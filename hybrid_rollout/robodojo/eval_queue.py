"""Strict multi-task evaluation on one eight-slot queue.

The dispatcher can evolve independently of an explicitly pinned episode source.
This preserves exact old runtime identities and completed receipts at handoff;
it never relaxes the evaluator's source/prompt/checkpoint resume checks.
"""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import tempfile

from .ssh_connection import identity_endpoint

from .eval_sequence import read_config, campaign_env, save_report
from .evaluation import read_panel, verify_assets
from .io import require, write_json


def execution_root(cfg, env):
    from .eval_snapshot import requested_source
    return requested_source(cfg, env)


def validate_queue(cfg):
    require(cfg.get('scheduling') == 'cross_task_queue_v1', 'Expected cross-task eval scheduling')
    require(all('task' in m for m in cfg['models']), 'Unified queue requires task panels')
    # Default queues retain one policy. A paired transfer queue explicitly
    # opts into mixed checkpoints, with separate identities and result roots.
    identities = {json.dumps({k:m.get(k) for k in ('checkpoint','policy_input','downsampling','prompt_profile')},
                             sort_keys=True) for m in cfg['models']}
    if cfg.get('experiment_kind') == 'paired_order_queue_v1':
        from .placement_order import VERSION as ORDER_VERSION
        from .prompt_profile import TRANSFER_BASELINE, GUIDED
        models = cfg['models']
        require(len(models) == 6 and len({m['experiment'] for m in models}) == 6 and
                {m.get('prompt_profile') for m in models} == {TRANSFER_BASELINE, GUIDED} and
                len({m['checkpoint'] for m in models}) == 2 and
                all(m.get('downsampling') == models[0].get('downsampling') for m in models),
                'Mixed order queue requires six distinct paired policy/order experiments')
        # Validate each policy's complete 1234/4321/2314 set using the original
        # contract. Only scheduling is mixed; neither policy is relaxed.
        for profile in (TRANSFER_BASELINE, GUIDED):
            validate_queue(dict(cfg, experiment_kind=ORDER_VERSION,
                                models=[m for m in models if m.get('prompt_profile') == profile]))
        return
    mixed = cfg.get('experiment_kind') == 'paired_transfer_queue_v1'
    if mixed:
        from .prompt_profile import TRANSFER_BASELINE, GUIDED
        models = cfg['models']
        require(len(models) == 2 and
                {m.get('prompt_profile') for m in models} == {TRANSFER_BASELINE, GUIDED} and
                len({m['experiment'] for m in models}) == 2 and
                len({m['checkpoint'] for m in models}) == 2 and
                all(m['task'] == 'solve_equation' and m['policy_input'] == 'task_subtask'
                    for m in models) and
                models[0]['cases'] == models[1]['cases'] and
                models[0]['downsampling'] == models[1]['downsampling'],
                'Mixed transfer requires two explicitly paired policies on identical cases')
    else:
        require(len(identities) == 1, 'Cross-task eval must use one policy/execution setting')
    from .placement_order import VERSION as ORDER_VERSION, ORDERS
    if cfg.get('experiment_kind') == ORDER_VERSION:
        require(len(cfg['models']) == 3 and
                [m.get('placement_order') for m in cfg['models']] == list(ORDERS) and
                all(m['task'] == 'arrange_largest_number' and m.get('cases') == [['standard', 0]]
                    and m['policy_input'] == 'task_subtask' for m in cfg['models']) and
                len({m['experiment'] for m in cfg['models']}) == 3,
                'Order queue requires three distinct order experiments on the same case')
    else:
        require(not any('placement_order' in m for m in cfg['models']), 'Missing order experiment kind')
        require(mixed or len({m['task'] for m in cfg['models']}) == len(cfg['models']), 'Duplicate task in eval queue')


def prepare_campaigns(cfg, env, prepared, root):
    from .dual_campaign import configure
    # Each worker runs one case via the legacy account-pool/batch subprocess.
    # Keep its non-policy validation setting compatible with the pinned old
    # evaluator (1..4); actual concurrency is len(slots), controlled by cfg.
    episode_cfg = dict(cfg, workers=min(cfg['workers'], 4))
    result = []
    for model, panel, path in prepared:
        child = campaign_env(env, episode_cfg, model, path)
        child.update(CODE_ROOT=str(root), PYTHONPATH=str(root),
                     ROLLOUT_EPISODE_ENTRYPOINT=str(root/'hybrid_rollout/robodojo/dual_machine_episode_entrypoint.sh'))
        configured, cases = configure(child)
        from .eval_exclusions import filter_cases
        cases = filter_cases(child['ROLLOUT_SHARED_ROOT'], model['experiment'], cases)
        result.append((configured, cases))
    return result


def queue_slots(cfg, campaign):
    from .parallel_collection import slots
    physical = slots(dict(campaign, ROLLOUT_PARALLEL_WORKERS=str(cfg['workers'])))
    # Only the dispatcher owns these eight slots. Subprocess validation does not
    # launch more workers, and must remain compatible with the pinned source.
    for slot in physical:
        slot['ROLLOUT_PARALLEL_WORKERS'] = campaign['ROLLOUT_PARALLEL_WORKERS']
    return physical


def attach_identities(campaigns, slots, env, fds):
    from .dual_campaign import runtime_identity
    from .regression_eval import runtime_settings
    identities = {}
    for child, _ in campaigns:
        # Never reuse another policy's checkpoint or prompt identity. Task-only
        # changes still share the expensive checks, then restore panel settings.
        policy = tuple(child.get(k) for k in ('CHECKPOINT', 'ROLLOUT_CONTROLLER_ROUTE',
            'ROLLOUT_PROMPT_PROFILE', 'ROLLOUT_ACTION_DOWNSAMPLING', 'ROLLOUT_DOWNSAMPLING_CONFIG'))
        hosts = {}
        for slot in slots:
            merged = dict(slot, **child)
            for name in ('ROLLOUT_SIM_HOST', 'ROLLOUT_SIM_SSH_PORT', 'ROLLOUT_SIM_SSH_USER',
                         'ROLLOUT_SIM_PYTHON', 'ROLLOUT_SIM_ROBODOJO_SOURCE', 'ROLLOUT_SIM_SSH_KEY'):
                if name in slot:
                    merged[name] = slot[name]
            host = identity_endpoint(merged)
            key = (policy, host)
            if key not in identities:
                identities[key] = runtime_identity(merged)
            hosts[host] = identities[key]
        base = next(iter(hosts.values()))
        require(all(value == base for value in hosts.values()),
                'Simulator hosts have different runtime identities')
        identity = dict(base, panel_sha256=child['ROLLOUT_EVAL_MANIFEST_SHA256'])
        identity.update(runtime_settings(child, read_panel(child['ROLLOUT_EVAL_MANIFEST'])))
        child['ROLLOUT_RESUME_IDENTITY'] = json.dumps(identity, sort_keys=True)
        child['ROLLOUT_RESOURCE_LOCK_FDS'] = ','.join(map(str, fds))
        # No upgrade/rewriting: every predecessor must match exactly.
        root = Path(child['ROLLOUT_SHARED_ROOT'])/'results'/child['ROLLOUT_EXPERIMENT_ID']
        paths = list(root.glob('_replicas/replica_*/attempt_*/batch.json'))
        paths += list(root.glob('_account_pool/run_*/pool_run.json'))
        for path in paths:
            prior = json.loads(path.read_text()).get('resume_identity')
            require(prior == identity, f'Eval identity changed; refusing to mix results: {path}')


def validate_expected_runtime(cfg, campaigns):
    for child,_ in campaigns:
        actual=json.loads(child['ROLLOUT_RESUME_IDENTITY'])
        for key,value in cfg.get('expected_runtime',{}).items():
            require(actual.get(key)==value, f'Prepared baseline runtime changed: {key}')


def select_retry_campaigns(campaigns, selections):
    """Filter dispatch only; retain the original panel and runtime identity."""
    if not selections:
        return campaigns
    wanted = set(selections)
    require(len(wanted) == len(selections), 'Duplicate retry selection')
    found = set()
    selected = []
    for child, cases in campaigns:
        kept = []
        for case in cases:
            key = child['ROLLOUT_EXPERIMENT_ID'] + '=' + case['case_id']
            if key in wanted:
                kept.append(case)
                found.add(key)
        if kept:
            selected.append((child, kept))
    require(found == wanted, 'Unknown or excluded retry selections: ' + str(sorted(wanted-found)))
    return selected


def select_retry_slots(physical, indices):
    if indices is None:
        return physical
    require(bool(indices) and len(set(indices)) == len(indices)
            and all(0 <= i < len(physical) for i in indices), 'Invalid physical slot selection')
    # Supervisor process indices must remain contiguous; physical ports/GPUs do not change.
    return [dict(physical[i], ROLLOUT_WORKER_ID=f'worker_{n}') for n, i in enumerate(indices)]


def run(cfg, env, *, dry_run=False, check_runtime=False,
        selections=None, retry_name=None, physical_slots=None):
    from .task_eval import freeze
    from .dual_campaign import acquire_resources, resource_keys
    from .parallel_collection import run_parallel, suite_jobs
    validate_queue(cfg)
    require(bool(selections) == bool(retry_name), 'Retry requires both --select and --retry-name')
    require(physical_slots is None or bool(retry_name), 'Physical slots require explicit retry selection')
    if retry_name:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', retry_name), 'Invalid retry name')
    if cfg.get('teacher'):
        teacher = cfg['teacher']
        for key,value in dict(ROLLOUT_MODEL=teacher['model'],
                ROLLOUT_REASONING_EFFORT=teacher['effort'],
                ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT=str(teacher['auto_compact_token_limit'])).items():
            require(env.get(key)==value, f'Pinned teacher requires {key}={value}')
        require(int(env.get('CODEX_IMAGE_MAX_EDGE','480'))==teacher['image_max_edge'],
                'Pinned teacher image size changed')
    from .eval_snapshot import experiment_source, bind_queue_config
    shared = Path(env['ROLLOUT_SHARED_ROOT'])
    output = shared/'evaluation'/cfg['name']
    queue_root = output
    if retry_name:
        output = output/'retries'/retry_name
        require((queue_root/'queue_config.json').is_file(), 'Retry requires an initialized parent eval queue')
    frozen = dict(config=cfg, teacher={k:env.get(k) for k in (
        'ROLLOUT_MODEL','ROLLOUT_REASONING_EFFORT','ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT')})
    if retry_name:
        frozen['retry_selection'] = dict(cases=selections, physical_slots=physical_slots)
    lock = None; fds = []
    try:
        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            # Share the original suite lock: a subset cannot overlap its parent queue.
            queue_root.mkdir(parents=True, exist_ok=True)
            lock = (queue_root/'queue.lock').open('a')
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('This eval queue is already running') from None
        with tempfile.TemporaryDirectory(prefix='robodojo-eval-queue-') as temporary:
            root = experiment_source(cfg, env, queue_root, dry_run=dry_run, temporary=temporary)
            bind_queue_config(queue_root, output, frozen, root, dry_run=dry_run, retry=bool(retry_name))
            prepared=[]
            for model in cfg['models']:
                path=shared/'eval_panels'/model['experiment']/'queue.json'
                expected=freeze(env['ROBODOJO_SOURCE'], model['task'], model['cases'])
                panel=read_panel(path) if path.exists() else expected
                require(panel == expected, 'Frozen task panel changed')
                verify_assets(panel, env['ROBODOJO_SOURCE'], verify_source=False)
                if dry_run:path=Path(temporary)/(model['experiment']+'.json')
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('x') as stream:json.dump(panel,stream,indent=2)
                prepared.append((model,panel,path))
            campaigns=select_retry_campaigns(prepare_campaigns(cfg,env,prepared,root), selections)
            slots=select_retry_slots(queue_slots(cfg,campaigns[0][0]), physical_slots)
            print(json.dumps(dict(event='eval_cross_task_queue_plan', output=str(output),
                workers=len(slots), planned_episodes=len(suite_jobs(campaigns)),
                execution_source=str(root), scheduling='round_robin_tasks_no_task_barrier',
                slots=[{k:e[k] for k in ('ROLLOUT_WORKER_ID','ROLLOUT_SIM_SSH_PORT','SIM_GPU',
                    'POLICY_PORT','SIM_PORT','ROLLOUT_SIM_REMOTE_PORT')} for e in slots],
                jobs=[dict(task=campaigns[i][0]['ROBODOJO_TASK'],case_id=job[1]['case_id'],
                           experiment=campaigns[i][0]['ROLLOUT_EXPERIMENT_ID'],
                           **({'placement_order':campaigns[i][0]['ROLLOUT_PLACEMENT_ORDER']}
                              if campaigns[i][0].get('ROLLOUT_PLACEMENT_ORDER') else {}))
                      for i,job in suite_jobs(campaigns)])),flush=True)
            if dry_run:
                if check_runtime:
                    attach_identities(campaigns,slots,env,[])
                    validate_expected_runtime(cfg,campaigns)
                    print(json.dumps(dict(event='eval_queue_runtime_verified', hosts=2,
                                          tasks=len(campaigns))),flush=True)
                return 0
            keys=[k for slot in slots for k in resource_keys(slot)]
            keys += [['campaign',e['ROLLOUT_EXPERIMENT_ID'],e['ROLLOUT_REPLICA_ID']] for e,_ in campaigns]
            keys += [['local_port',os.uname().nodename,e[k]] for e in slots for k in ('POLICY_PORT','SIM_PORT')]
            fds=acquire_resources(shared/'private_runtime/dual_locks',keys,wait=True)
            campaigns=select_retry_campaigns(prepare_campaigns(cfg,env,prepared,root), selections)
            slots=select_retry_slots(queue_slots(cfg,campaigns[0][0]), physical_slots)
            print(json.dumps(dict(event='eval_queue_checking_runtime_identities')),flush=True)
            attach_identities(campaigns,slots,env,fds)
            validate_expected_runtime(cfg,campaigns)
            slots=select_retry_slots(queue_slots(cfg,campaigns[0][0]), physical_slots)
            try:
                code=run_parallel(slots,campaigns[0][1],campaigns=campaigns,output=output/'queue')
            finally:
                reports=[save_report(shared,model) for model in cfg['models']]
                write_json(output/'summary.json',dict(planned=sum(r['planned'] for r in reports),
                    original_planned=sum(r.get('original_planned',r['planned']) for r in reports),
                    excluded=sum(r.get('excluded',0) for r in reports),
                    completed=sum(r['completed'] for r in reports),successes=sum(r['successes'] for r in reports),
                    failures=sum(r['failures'] for r in reports),tasks=reports))
            return code
    finally:
        for fd in fds:os.close(fd)
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--check-runtime',action='store_true',
                   help='Read-only SSH/source/checkpoint identity checks; requires --dry-run')
    p.add_argument('--select', action='append', help='Retry only experiment=case_id; repeatable')
    p.add_argument('--retry-name', help='Separate dispatcher records; episode archives stay in the original experiment')
    p.add_argument('--physical-slots', help='Comma-separated physical slots from the original queue, e.g. 0,1')
    args=p.parse_args()
    if args.check_runtime and not args.dry_run:p.error('--check-runtime requires --dry-run')
    return run(read_config(args.config),dict(os.environ),dry_run=args.dry_run,
               check_runtime=args.check_runtime, selections=args.select, retry_name=args.retry_name,
               physical_slots=[int(x) for x in args.physical_slots.split(',')]
               if args.physical_slots else None)


if __name__=='__main__':raise SystemExit(main())

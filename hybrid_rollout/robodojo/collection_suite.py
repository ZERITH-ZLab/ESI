"""Task-specific 20-case training panels and a resumable, cross-task pilot queue.

Uses the existing collection/account/parallel engine. No simulator imports,
model calls or mounted writes during --dry-run.
"""
import argparse
import ast
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from .ssh_connection import identity_endpoint

from .collection import SCHEMA
from .evaluation import canonical_sha256, file_record, freeze_panel, layout_files, read_panel, verify_assets
from .io import require, write_json
from .task_success_conditions import render_task_success_conditions

TASKS = (
    'pack_objects_into_box', 'arrange_largest_number', 'sort_nesting_dolls_by_size',
    'classify_objects', 'fill_pen_holder', 'organize_table', 'imitate_sorting_sequence',
    'stack_blocks_by_language', 'classify_objects_by_language', 'store_tools_in_toolbox',
    'pick_from_conveyor_by_image', 'sweep_blocks',
)
# Append new tasks to preserve existing replica identities.
RANDOM_TASKS = frozenset((*TASKS[:3], 'sweep_blocks'))


def task_specs(task):
    require(task in TASKS, 'Task not in this pilot suite')
    variants = ('standard', 'random') if task in RANDOM_TASKS else ('standard',)
    for index, (variant, layout) in enumerate((v, n) for v in variants for n in range(20//len(variants))):
        yield dict(case_id=f'{task}__{variant}__g0__l{layout}', task=task,
            runtime_task=task + ('_random' if variant == 'random' else ''), variant=variant,
            replica_id=TASKS.index(task), rollout_index=index, eval_seed=0,
            layout_id=layout, reset_seed=layout, simulator_initial_seed=0, policy_rng_seed=0)


def pilot_cases(panel):
    count = 2 if panel['collection']['task'] in RANDOM_TASKS else 3
    return [c for c in panel['cases'] if c['layout_id'] < count]


def selected_cases(panel, cfg):
    selection = cfg.get('case_selection', 'pilot')
    require(selection in ('pilot', 'full20'), 'case_selection must be pilot or full20')
    return list(panel['cases']) if selection == 'full20' else pilot_cases(panel)


def native_limit(source, task):
    path = Path(source)/'task/RoboDojo/tasks'/f'{task}.py'
    tree = ast.parse(path.read_text())
    values = [node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == 'self' and t.attr == 'step_lim' for t in node.targets)
        and isinstance(node.value, ast.Constant) and type(node.value.value) is int]
    require(len(values) == 1 and values[0] > 0, f'Cannot statically identify native step limit: {task}')
    return values[0]


def validate_task_panel(panel):
    require(panel.get('schema') == SCHEMA and panel.get('split') == 'train', 'Not a training collection panel')
    require(panel['panel_sha256'] == canonical_sha256({k:v for k,v in panel.items() if k != 'panel_sha256'}), 'Collection panel hash changed')
    cfg = panel['collection']
    require(cfg.get('kind') == 'task20_v1' and cfg.get('task') in TASKS, 'Invalid task20 protocol')
    require(cfg.get('repeats') == 1 and cfg.get('order') == 'repeat_major'
            and cfg.get('policy_rng_seed') == 0 and cfg.get('eval_seed') == 0
            and panel.get('eval_seed') == 0 and type(cfg.get('max_seconds')) is int and cfg['max_seconds'] > 0, 'Task20 protocol changed')
    require(type(cfg.get('native_step_limit')) is int and cfg['native_step_limit'] > 0
            and type(cfg.get('step_limit')) is int and cfg['step_limit'] >= cfg['native_step_limit'], 'Invalid collection step budget')
    require(len(panel['cases']) == 20, 'Expected a 20-case parent panel')
    for actual, expected in zip(panel['cases'], task_specs(cfg['task'])):
        require(all(actual.get(k) == v for k,v in expected.items()), 'Task20 case identity/order changed')
        support = actual.get('native_support_trajectories', [])
        expected_support = ([f'Assets/Traj/RoboDojo/imitate_sorting_sequence/0/{actual["layout_id"]}.pkl']
                            if cfg['task'] == 'imitate_sorting_sequence' else [])
        require([r['path'] for r in support] == expected_support, 'Support trajectory mapping changed')
        for record in [actual['layout'], *support]:
            path = Path(record['path'])
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe asset path')
    return panel


def freeze_task(source, task, budget, template=None, max_seconds=10800):
    source = Path(source).resolve()
    panel = deepcopy(template) if template else freeze_panel(source, 'task20_inventory', 0)
    has_random = bool(layout_files(source, task+'_random', 0))
    require(has_random == (task in RANDOM_TASKS), f'Variant inventory changed for {task}; review panel')
    limit = native_limit(source, task)
    if has_random:
        require(native_limit(source, task+'_random') == limit, 'Variant native budgets differ')
    numerator, denominator = budget
    extended = (limit*numerator + denominator-1)//denominator
    panel.update(panel_id=f'{task}_train20_g0_v1', schema=SCHEMA, split='train',
        scope='training_collection_not_official_evaluation',
        selection='First 20 standard, or first 10 standard and first 10 random; pilot uses first 3 or 2+2',
        collection=dict(kind='task20_v1', task=task, eval_seed=0, repeats=1, order='repeat_major',
                        native_step_limit=limit, step_limit=extended, max_seconds=max_seconds, policy_rng_seed=0), cases=[])
    files = {v:layout_files(source, task+('_random' if v == 'random' else ''), 0)
             for v in (('standard','random') if has_random else ('standard',))}
    for spec in task_specs(task):
        paths = files[spec['variant']]
        require(spec['layout_id'] < len(paths), f'Missing layout: {spec["case_id"]}')
        support = ([file_record(source, source/'Assets/Traj/RoboDojo/imitate_sorting_sequence/0'/f'{spec["layout_id"]}.pkl')]
                   if task == 'imitate_sorting_sequence' else [])
        # Exact runtime-task section; unknown tasks fail instead of borrowing digit rules.
        brief = render_task_success_conditions(spec['runtime_task'])
        require(task == 'arrange_largest_number' or 'Flip recovery:' not in brief, 'Digit prompt leaked')
        panel['cases'].append(dict(spec, layout=file_record(source, paths[spec['layout_id']]),
                                  native_support_trajectories=support))
    panel['panel_sha256'] = canonical_sha256({k:v for k,v in panel.items() if k != 'panel_sha256'})
    return validate_task_panel(panel)


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    from .prompt_evolution.config import settings as evolution_settings
    evolution_settings(cfg.get('prompt_evolution'))
    if cfg.get('prompt_bundle'):
        from .prompt_evolution.bundle import load
        bundle = load(cfg['prompt_bundle'])
        require(cfg.get('prompt_bundle_sha256') == bundle['sha256'], 'Pinned prompt bundle hash changed')
    require(cfg.get('schema') == 'robodojo.collection_suite.v1', 'Unknown collection suite config')
    require(cfg.get('case_selection', 'pilot') in ('pilot', 'full20'),
            'case_selection must be pilot or full20')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', cfg['name']), 'Invalid suite name')
    require(cfg.get('controller_route', 'auto_infer_v1') in ('auto_infer_v1', 'subtask_feedforward_v1'),
            'Unsupported collection controller_route')
    tasks = cfg.get('tasks')
    require(isinstance(tasks, list) and bool(tasks)
            and all(isinstance(task, str) and task in TASKS for task in tasks)
            and len(set(tasks)) == len(tasks),
            'tasks must be a nonempty, duplicate-free subset of the supported base tasks')
    require(type(cfg['workers']) is int and cfg['workers'] >= 1, 'Workers must be positive; capacity is checked against deployment slots')
    require(len(cfg['budget_multiplier']) == 2 and all(type(n) is int and n > 0 for n in cfg['budget_multiplier'])
            and cfg['budget_multiplier'][0] >= cfg['budget_multiplier'][1], 'Budget multiplier must be >=1')
    require(isinstance(cfg['model'], str) and bool(cfg['model'].strip()), 'Collection model is required')
    require(cfg['reasoning_effort'] in ('low', 'medium', 'high', 'xhigh', 'max', 'ultra'), 'Invalid reasoning effort')
    for key, default in [('max_seconds',10800),('auto_compact_token_limit',96000),('image_max_edge',480)]:
        require(type(cfg.get(key,default)) is int and cfg.get(key,default)>0, f'Invalid {key}')
    if 'pi_force_hold' in cfg:
        hold = cfg['pi_force_hold']
        require(isinstance(hold, dict) and set(hold) == {'enabled', 'config'}
                and type(hold['enabled']) is bool and isinstance(hold['config'], dict),
                'pi_force_hold requires enabled (boolean) and config (object)')
        from .robodojo_server.pi_force_hold import configured, validate_tightening_context
        configured(hold['config'])
        if hold['enabled']:
            validate_tightening_context(hold['config'],
                controller_route=cfg.get('controller_route', 'auto_infer_v1'))
    ckpt = Path(cfg['checkpoint'])
    require(ckpt.is_absolute() and (ckpt/'params').is_dir()
            and (ckpt/'assets/arx_x5_sim/norm_stats.json').is_file(), 'Missing base checkpoint/normalization')
    return cfg


def campaign_env(env, cfg, panel, panel_path, control_path):
    child = dict(env)
    from .prompt_evolution.bundle import ENV, load
    child.pop(ENV, None)
    if cfg.get('prompt_bundle'):
        bundle = load(cfg['prompt_bundle'])
        require(cfg.get('prompt_bundle_sha256') == bundle['sha256'], 'Pinned prompt bundle hash changed')
        child[ENV] = json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))
    for key in ('ROLLOUT_CASE_ID', 'ROLLOUT_COLLECTION_REPEAT_ID', 'ROLLOUT_COLLECTION_JOB_INDEX',
                'ROLLOUT_POLICY_RNG_SEED', 'ROLLOUT_SEED_BASE', 'ROLLOUT_RERUN_REFERENCE'):
        child.pop(key, None)
    task = panel['collection']['task']; name = cfg['name']+'__'+task
    root = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/name
    prior = list(root.glob('_replicas/replica_*/attempt_*')) + list(root.glob('_account_pool/run_*/pool_run.json'))
    chosen = selected_cases(panel, cfg)
    child.update(ROBODOJO_TASK=task, ROLLOUT_COUNT=str(len(chosen)),
        ROLLOUT_CASE_IDS=','.join(c['case_id'] for c in chosen),
        ROLLOUT_EXPERIMENT_ID=name, ROLLOUT_EVAL_MANIFEST=str(panel_path),
        ROLLOUT_RUN_MODE='resume' if prior else 'fresh', ROLLOUT_ATTEMPT='auto',
        ROLLOUT_CONTROLLER_ROUTE=cfg.get('controller_route', 'auto_infer_v1'), ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt',
        ROLLOUT_STRICT_EVAL='0', ROLLOUT_ENABLE_SCENE_QUERY='1', ROLLOUT_ENABLE_MOTION_TOOLS='1',
        ROLLOUT_ENABLE_TASK_MEMORY='1', ROLLOUT_ENABLE_TEACHER_ABANDON='1',
        ROLLOUT_MODEL=cfg['model'], ROLLOUT_REASONING_EFFORT=cfg['reasoning_effort'],
        ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT=str(cfg.get('auto_compact_token_limit',96000)),
        CODEX_IMAGE_MAX_EDGE=str(cfg.get('image_max_edge',480)), MAX_DECISIONS='0', CODEX_MAX_TOTAL_TOKENS='0',
        ROLLOUT_COLLECTION_STEP_LIMIT=str(panel['collection']['step_limit']), ROLLOUT_MAX_SECONDS=str(panel['collection']['max_seconds']),
        ROLLOUT_ALLOW_SOURCE_UPGRADE='0', ROLLOUT_PI05_COLLISION_SHADOW='0',
        ROLLOUT_ACCOUNT_POOL_FILE=cfg['account_pool_file'], CHECKPOINT=cfg['checkpoint'],
        ROLLOUT_WORKER_CONTROL_FILE=str(control_path), ROLLOUT_PARALLEL_WORKERS=str(cfg['workers']))
    if 'pi_force_hold' in cfg:
        from dataclasses import asdict
        from .robodojo_server.pi_force_hold import configured
        hold = cfg['pi_force_hold']
        child.update(ROBODOJO_GRIPPER_CONTACT_ENABLED='1',
                     ROBODOJO_PI_FORCE_HOLD_ENABLED='1' if hold['enabled'] else '0',
                     ROBODOJO_PI_FORCE_HOLD_CONFIG=json.dumps(asdict(configured(hold['config'])),
                                                             sort_keys=True))
    return child


def run_unified(cfg, env, prepared, output, *, dry_run=False, allow_source_upgrade=False):
    from .dual_campaign import configure, runtime_identity, acquire_resources, resource_keys
    from .parallel_collection import slots, run_parallel, suite_jobs, SLOT_KEYS

    def campaigns():
        result = [configure(campaign_env(env, cfg, panel, path, control))
                  for panel, path, control in prepared]
        for e, _ in result:
            if allow_source_upgrade and e['ROLLOUT_RUN_MODE'] == 'resume':
                e['ROLLOUT_ALLOW_SOURCE_UPGRADE'] = '1'
        return result

    configured = campaigns()
    environments = slots(configured[0][0])
    keys = [key for slot in environments for key in resource_keys(slot)]
    keys += [['campaign', e['ROLLOUT_EXPERIMENT_ID'], e['ROLLOUT_REPLICA_ID']] for e,_ in configured]
    keys += [['local_port', os.uname().nodename, e[k]] for e in environments for k in ('POLICY_PORT','SIM_PORT')]
    print(json.dumps(dict(event='collection_suite_queue', jobs=[dict(task=configured[t][0]['ROBODOJO_TASK'],
        case_id=job[1]['case_id']) for t,job in suite_jobs(configured)],
        worker_control=str(prepared[0][2]), slots=[{k:e[k] for k in SLOT_KEYS if k in e
            and k not in ('CODE_ROOT','PYTHONPATH','ROLLOUT_RESOURCE_LOCK_FDS')} for e in environments]),
        ensure_ascii=False), flush=True)
    if dry_run:
        return 0
    fds = acquire_resources(Path(env['ROLLOUT_SHARED_ROOT'])/'private_runtime/dual_locks', keys, wait=True)
    try:
        configured = campaigns()  # Recheck previous attempts after waiting for resources.
        for e, _ in configured:
            print(json.dumps(dict(event='collection_suite_identity_check', task=e['ROBODOJO_TASK'])), flush=True)
            identities = {}
            for slot in slots(e):
                host = identity_endpoint(slot)
                if host not in identities:
                    identities[host] = runtime_identity(slot)
            values = list(identities.values())
            require(all(v == values[0] for v in values), 'Both simulator hosts must have identical native sources')
            e['ROLLOUT_RESUME_IDENTITY'] = json.dumps(values[0], sort_keys=True)
            e['ROLLOUT_RESOURCE_LOCK_FDS'] = ','.join(map(str, fds))
            from .resume_upgrade import prepare
            prepare(e)
        environments = slots(configured[0][0])
        write_json(output/'status.json', dict(state='running', planned_episodes=len(suite_jobs(configured)),
            experiments=[e['ROLLOUT_EXPERIMENT_ID'] for e,_ in configured], queue_root=str(output/'queue')))
        code = run_parallel(environments, configured[0][1], campaigns=configured, output=output/'queue')
        # Preserve the supervisor's deferred cases and circuit-breaker reason.
        # Exit 0 means dispatch finished, not necessarily that every case was collected.
        final_status = json.loads((output/'status.json').read_text())
        if final_status.get('state') not in ('completed_with_pending_recollection',
                                           'dispatch_paused_connection_errors'):
            final_status['state'] = ('completed' if code == 0 else 'completed_with_timeouts'
                                     if code == 124 else 'failed')
        final_status.update(returncode=code, queue_root=str(output/'queue'))
        write_json(output/'status.json', final_status)
        return code
    except BaseException:
        write_json(output/'status.json', dict(state='stopped', queue_root=str(output/'queue'),
            resume='Rerun the same command; verified completed episodes are skipped'))
        raise
    finally:
        for fd in fds:
            os.close(fd)


def run(cfg, env, *, dry_run=False, allow_source_upgrade=False):
    require(env.get('ROLLOUT_PROMPT_EVOLUTION', '0') in ('0', '1'),
            'ROLLOUT_PROMPT_EVOLUTION must be 0 or 1')
    shared = Path(env['ROLLOUT_SHARED_ROOT']); source = Path(env['ROBODOJO_SOURCE'])
    output = shared/'collection_suites'/cfg['name']
    from .account_pool import read_config as pool_config
    pool = pool_config(cfg['account_pool_file'])
    template = freeze_panel(source, 'task20_inventory', 0)
    panels = [freeze_task(source, task, cfg['budget_multiplier'], template, cfg.get('max_seconds',10800)) for task in cfg['tasks']]
    total = sum(len(selected_cases(p, cfg)) for p in panels)
    print(json.dumps(dict(event='collection_suite_plan', output=str(output), episodes=total,
        workers=cfg['workers'], profiles=pool['profiles'], route=cfg.get('controller_route', 'auto_infer_v1'), checkpoint=cfg['checkpoint'],
        case_selection=cfg.get('case_selection', 'pilot'),
        scheduling='one cross-task queue; round-robin tasks; no completion barrier',
        pi_force_hold=cfg.get('pi_force_hold'),
        tasks=[dict(task=p['collection']['task'], episodes=len(selected_cases(p, cfg)),
                    native_steps=p['collection']['native_step_limit'], collection_steps=p['collection']['step_limit'],
                    case_ids=[c['case_id'] for c in selected_cases(p, cfg)]) for p in panels]), ensure_ascii=False), flush=True)
    lock = None
    try:
        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            lock = (output/'suite.lock').open('a')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = output/'config.json'
            if frozen.exists():
                require(json.loads(frozen.read_text()) == cfg, 'Suite config changed; use a new name')
            else:
                with frozen.open('x') as f:json.dump(cfg, f, indent=2)
        with tempfile.TemporaryDirectory(prefix='robodojo-collection-suite-') as tmp:
            # Freeze/verify every task before starting the first episode.
            prepared=[]
            for panel in panels:
                task=panel['collection']['task'];path=output/'panels'/f'{task}.json'
                if path.exists():
                    old=read_panel(path)
                    require(old['collection']==panel['collection'] and old['cases']==panel['cases'], 'Frozen task panel changed')
                    panel=old
                verify_assets(panel,source,verify_source=False)
                if dry_run:path=Path(tmp)/f'{task}.json'
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('x') as f:json.dump(panel,f,indent=2)
                control=(Path(tmp) if dry_run else output)/'worker_control.json'
                if not control.exists():write_json(control,dict(desired_workers=cfg['workers']))
                prepared.append((panel,path,control))
            code = run_unified(cfg, env, prepared, output, dry_run=dry_run,
                               allow_source_upgrade=allow_source_upgrade)
            if not dry_run:
                from .prompt_evolution.pipeline import after_collection
                after_collection(cfg, env, output, code)
            return code
    finally:
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).with_name('collection_suite.pilot10.json')))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--allow-source-upgrade', action='store_true',
                   help='Explicitly accept prior collection receipts after a code-only update; all other identities must match')
    args=p.parse_args()
    raise SystemExit(run(read_config(args.config),dict(os.environ),dry_run=args.dry_run,
                        allow_source_upgrade=args.allow_source_upgrade))


if __name__=='__main__':main()

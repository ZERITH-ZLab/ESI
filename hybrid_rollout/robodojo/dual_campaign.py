"""Resolve launch_dual.sh settings and lease one serial two-host campaign."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from .ssh_connection import identity_endpoint, simulator_target

from .deployment import digest, native_identity, source_identity
from .evaluation import read_panel, replica_cases
from .io import sha256
from .settings import MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
from .model_config import model_selection


def configure(env):
    env = dict(env)
    from .downsampling import configuration
    configuration(env)
    model, effort = model_selection(env)
    env['ROLLOUT_MODEL'], env['ROLLOUT_REASONING_EFFORT'] = model, effort
    shadow = env.setdefault('ROLLOUT_PI05_COLLISION_SHADOW', '0')
    if shadow not in ('0', '1'):
        raise ValueError('ROLLOUT_PI05_COLLISION_SHADOW must be 0 or 1')
    motion = env.setdefault('ROLLOUT_ENABLE_MOTION_TOOLS', '0')
    if motion not in ('0', '1'):
        raise ValueError('ROLLOUT_ENABLE_MOTION_TOOLS must be 0 or 1')
    if motion == '1' and (env.get('ROLLOUT_CONTROLLER_ROUTE', 'baseline') not in ('auto_infer_v1', 'subtask_feedforward_v1')
                         or env.get('ROLLOUT_EVALUATION_METHOD') != 'pi05_plus_gpt'):
        raise ValueError('Motion tools require auto_infer_v1/subtask_feedforward_v1 + pi05_plus_gpt')
    enabled = env.setdefault('ROLLOUT_ENABLE_SCENE_QUERY', '0')
    if enabled not in ('0', '1'):
        raise ValueError('ROLLOUT_ENABLE_SCENE_QUERY must be 0 or 1')
    if enabled == '1' and (env.get('ROLLOUT_CONTROLLER_ROUTE', 'baseline') not in ('auto_infer_v1', 'subtask_feedforward_v1')
                            or env.get('ROLLOUT_EVALUATION_METHOD') != 'pi05_plus_gpt'):
        raise ValueError('Scene query requires auto_infer_v1/subtask_feedforward_v1 + pi05_plus_gpt')
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    from .collection import runtime_settings
    collection = runtime_settings(env, panel)
    from .regression_eval import runtime_settings as eval_settings
    eval_settings(env, panel)
    task = env['ROBODOJO_TASK']
    replicas = {c['replica_id'] for c in panel['cases'] if c['task'] == task}
    if len(replicas) != 1:
        raise ValueError('Task not in panel; use the base task name, not *_random')
    env['ROLLOUT_REPLICA_ID'] = str(replicas.pop())
    env['ROLLOUT_EVAL_MANIFEST_SHA256'] = panel['panel_sha256']
    cases = replica_cases(panel, task, env['ROLLOUT_REPLICA_ID'])
    chosen = env.get('ROLLOUT_CASE_IDS', '')
    if chosen:
        ids = [x.strip() for x in chosen.split(',')]
        if len(ids) != len(set(ids)) or not set(ids) <= {c['case_id'] for c in cases}:
            raise ValueError('Duplicate or unknown case ID for this task')
        cases = [next(c for c in cases if c['case_id'] == name) for name in ids]
        env['ROLLOUT_CASE_IDS'] = ','.join(ids)
        env['ROLLOUT_COUNT'] = str(len(cases))
    else:
        count = int(env['ROLLOUT_COUNT'])
        if not 1 <= count <= len(cases):
            raise ValueError(f'COUNT must be 1..{len(cases)}')
        cases = cases[:count]
    if collection and not chosen and cases != panel['cases']:
        raise ValueError('Collection COUNT must select all 20 cases; use explicit CASE_IDS for a subset')
    experiment = env['ROLLOUT_EXPERIMENT_ID']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', experiment):
        raise ValueError('Invalid experiment ID')
    base = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/experiment/'_replicas'/f'replica_{env["ROLLOUT_REPLICA_ID"]}'
    mode = env['ROLLOUT_RUN_MODE']
    if mode not in {'fresh', 'resume', 'rerun'}:
        raise ValueError('RUN_MODE must be fresh, resume or rerun')
    if collection and mode == 'rerun':
        raise ValueError('Collection uses fresh/resume only; use a new experiment for a new collection')
    attempts = list(base.glob('attempt_*'))
    pool_runs = list((base.parent.parent/'_account_pool').glob('run_*/pool_run.json'))
    # Even an explicit rerun must not mix teacher identities under one experiment.
    for previous in base.parent.glob('replica_*/attempt_*/batch.json'):
        identity = json.loads(previous.read_text()).get('resume_identity') or {}
        if identity and (identity.get('model'), identity.get('effort')) != (model, effort):
            raise ValueError('Experiment model/effort changed; use a new EXPERIMENT_ID')
    if collection:
        expected_selection = collection_selection_identity(env, panel)
        previous_receipts = list(base.parent.glob('replica_*/attempt_*/batch.json')) + pool_runs
        previous_receipts += list((base.parent.parent/'_parallel').glob('run_*/status.json'))
        for previous in previous_receipts:
            identity = json.loads(previous.read_text()).get('resume_identity') or {}
            if identity.get('selected_case_ids') != expected_selection.get('selected_case_ids'):
                raise ValueError('Experiment case selection changed; use a new EXPERIMENT_ID')
    if mode == 'fresh' and (attempts or pool_runs):
        raise ValueError('Experiment already exists: choose resume/rerun or a new experiment ID')
    if mode == 'resume' and not (attempts or pool_runs):
        raise ValueError('No prior attempt to resume; choose fresh')
    attempt = env['ROLLOUT_ATTEMPT']
    if attempt == 'auto':
        numbers = [int(p.name[8:]) for p in attempts if p.name[8:].isdigit()]
        attempt = str(max(numbers, default=-1) + 1)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt) or (base/f'attempt_{attempt}').exists():
        raise ValueError('Attempt already exists or is invalid; use auto or a fresh identifier')
    env['ROLLOUT_ATTEMPT'] = attempt
    env['ROLLOUT_SKIP_COMPLETED_CASES'] = '1' if mode == 'resume' else '0'
    return env, cases


def collection_selection_identity(env, panel):
    """Only explicit non-default selections extend the existing resume schema."""
    chosen = env.get('ROLLOUT_CASE_IDS', '')
    ids = [value.strip() for value in chosen.split(',')] if chosen else []
    if ids and ids != [case['case_id'] for case in panel['cases']]:
        return {'selected_case_ids': ids}
    return {}


def gripper_runtime_identity(env):
    """Record explicit cross-SSH settings without changing legacy identities."""
    keys = ('ROBODOJO_GRIPPER_CONTACT_ENABLED', 'ROBODOJO_PI_FORCE_HOLD_ENABLED',
            'ROBODOJO_PI_FORCE_HOLD_CONFIG')
    if not any(env.get(key) for key in keys):
        return {}
    contact = env.get(keys[0], '1')
    hold = env.get(keys[1], '0')
    if contact not in ('0', '1') or hold not in ('0', '1'):
        raise ValueError('Gripper feature flags must be 0 or 1')
    if hold == '1' and contact != '1':
        raise ValueError('Pi force hold requires gripper contact')
    from dataclasses import asdict
    from .robodojo_server.pi_force_hold import configured, execution_version, validate_tightening_context
    # Compare the same JSON representation used in saved/leased identities.
    # allowed_object_paths is a tuple in ForceHoldConfig, but a list on disk.
    config = json.loads(json.dumps(asdict(configured(json.loads(env.get(keys[2]) or '{}')))))
    if hold == '1':
        validate_tightening_context(config, controller_route=env.get('ROLLOUT_CONTROLLER_ROUTE'))
    return dict(gripper_execution=dict(contact_enabled=contact == '1',
        pi_force_hold_enabled=hold == '1', version=execution_version(config), config=config))


def runtime_identity(env):
    model, effort = model_selection(env)
    from .pi05_server.checkpoint import checkpoint_identity
    # Execute only this read-only fingerprint helper remotely; no repository sync required.
    script = ('import hashlib,json,subprocess,sys\nfrom pathlib import Path\n' +
              inspect.getsource(sha256) + '\n' + inspect.getsource(digest) + '\n' +
              inspect.getsource(native_identity) + '\nprint(native_identity(sys.argv[1]))\n')
    command = shlex.join([env['ROLLOUT_SIM_PYTHON'], '-', env['ROLLOUT_SIM_ROBODOJO_SOURCE']])
    remote = subprocess.check_output(['ssh', '-i', env['ROLLOUT_SIM_SSH_KEY'],
        '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-p', env['ROLLOUT_SIM_SSH_PORT'], simulator_target(env), command],
        input=script, text=True, timeout=120).strip()
    if not re.fullmatch('[0-9a-f]{64}', remote):
        raise ValueError('Invalid remote source fingerprint')
    identity = dict(**controller_route_identity(env), schema='robodojo.dual.resume.v1', panel_sha256=env['ROLLOUT_EVAL_MANIFEST_SHA256'],
        evaluation_method=env['ROLLOUT_EVALUATION_METHOD'], model=model, effort=effort,
        provider=PROVIDER, checkpoint_sha256=checkpoint_identity(env['CHECKPOINT'])['checkpoint_sha256'],
        source_sha256=source_identity(env['CODE_ROOT']), native_a100=native_identity(env['ROBODOJO_SOURCE']),
        native_4090=remote, openpi=native_identity(env['OPENPI_SOURCE']),
        codex_version=subprocess.check_output([env['CODEX_BIN'], '--version'], text=True).strip(),
        max_decisions=int(env['MAX_DECISIONS']), max_seconds=int(env['ROLLOUT_MAX_SECONDS']),
        image_max_edge=int(env.get('CODEX_IMAGE_MAX_EDGE', DEFAULT_CODEX_IMAGE_MAX_EDGE)),
        token_limit=int(env.get('CODEX_MAX_TOTAL_TOKENS', '0')))
    if env.get('ROLLOUT_PI05_COLLISION_SHADOW', '0') == '1':
        identity['pi05_collision_shadow'] = 'sampled_spheres_v1'
    if env.get('ROLLOUT_ENABLE_SCENE_QUERY') == '1':
        identity['information_access'] = 'role_query_v2'
        identity['scene_query_threshold'] = 2
    if env.get('ROLLOUT_ENABLE_MOTION_TOOLS') == '1':
        identity['motion_executor'] = 'link6_relaxed_quintic_ack_soft_boundary_v1'
    identity.update(gripper_runtime_identity(env))
    from .downsampling import identity as downsampling_identity
    identity.update(downsampling_identity(env))
    if env.get('ROBODOJO_PI_FORCE_HOLD_ENABLED') == '1':
        from .robodojo_server.pi_force_hold import validate_tightening_context
        validate_tightening_context(json.loads(env.get('ROBODOJO_PI_FORCE_HOLD_CONFIG') or '{}'),
            controller_route=env.get('ROLLOUT_CONTROLLER_ROUTE'),
            checkpoint_sha256=identity['checkpoint_sha256'])
    from .collection import runtime_settings
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    identity.update(runtime_settings(env, panel))
    from .regression_eval import runtime_settings as eval_settings
    identity.update(eval_settings(env, panel))
    if 'collection_protocol' in identity:
        identity.update(collection_selection_identity(env, panel))
    if env.get('ROLLOUT_WORKER_CONTROL_FILE'):
        identity['collection_scheduler'] = 'dynamic_workers_v1'
    elif env.get('ROLLOUT_PARALLEL_WORKERS', '1') != '1':
        if identity.get('eval_protocol'):
            identity['eval_scheduler'] = 'fixed_workers_v1'
        else:
            identity['collection_scheduler'] = 'parallel2_dynamic_v1'
    return identity


def controller_route_identity(env):
    route = env.get('ROLLOUT_CONTROLLER_ROUTE', 'baseline')
    from .skill_auto_infer.task_memory import enabled, VERSION as MEMORY_VERSION
    task_memory = enabled(env)
    from .skill_auto_infer.abandon import enabled as abandon_enabled, VERSION as ABANDON_VERSION
    abandon = abandon_enabled(env)
    if route not in ('baseline', 'auto_infer_v1', 'subtask_feedforward_v1'):
        raise ValueError('Unknown ROLLOUT_CONTROLLER_ROUTE')
    # Preserve the baseline identity schema; optimized runs cannot match it.
    if route == 'baseline':
        return {}
    from .skill_auto_infer.memory import compact_limit, VERSION
    result = {'controller_route': route, 'decision_history_version': VERSION,
              'model_auto_compact_token_limit': compact_limit(env)}
    from .prompt_evolution.bundle import from_env
    guidance = from_env(env)
    if guidance is not None:
        result['guidance_bundle_sha256'] = guidance['sha256']
    if route == 'subtask_feedforward_v1':
        result['conditioning_version'] = 'subtask_feedforward_v1'
    if task_memory:
        result['task_memory'] = dict(version=MEMORY_VERSION, recent_summaries=4,
                                    high_correction_ratio_summaries=4, min_correction_streak=60, max_words=999,
                                   scope='same_experiment_task_and_runtime_identity')
    if abandon:
        result['teacher_abandon'] = dict(version=ABANDON_VERSION, correction_streak_steps=120,
                                        latched_for_episode=True)
    return result


def resource_keys(env):
    """Separate machines behind one public IP; reserve GPU and port independently.

    Campaign lock has one supervisor owner; its workers share an atomic allocator.
    Repeated GPU keys are deduplicated only inside that supervisor's fixed slots.
    """
    host = [env['ROLLOUT_SIM_HOST'], env['ROLLOUT_SIM_SSH_PORT']]
    return [
        ['sim_port', *host, env['ROLLOUT_SIM_REMOTE_PORT']],
        ['sim_gpu', *host, env.get('SIM_GPU', '0')],
        ['campaign', env['ROLLOUT_EXPERIMENT_ID'], env['ROLLOUT_REPLICA_ID']],
    ]


def acquire_resources(lock_root, keys, *, wait=False, poll_seconds=5):
    """Acquire the entire set or release it before waiting; never hold partial locks."""
    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted({digest(key): key for key in keys}.items())
    last_message = None
    while True:
        acquired = []
        try:
            for name, key in ordered:
                fd = os.open(lock_root/(name+'.lock'), os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    os.close(fd)
                    raise
                acquired.append(fd)
                os.set_inheritable(fd, True)
            return acquired
        except BaseException as exc:
            for fd in acquired:
                os.close(fd)
            if not isinstance(exc, BlockingIOError):
                raise
            if not wait:
                raise SystemExit(f'Resource already in use: {key}') from exc
            now = time.monotonic()
            if last_message is None or now - last_message >= 30:
                print(json.dumps(dict(event='eval_waiting_for_resources', resource=key,
                    message='Existing campaign is still running; no episodes launched')), flush=True)
                last_message = now
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print selection only; no SSH, login, writes or model calls')
    parser.add_argument('--wait-for-resources', action='store_true',
                        help='Wait for the running campaign to release simulator GPUs and ports')
    args = parser.parse_args()
    env, cases = configure(os.environ)
    from .parallel_collection import slots, run_parallel, read_desired
    environments = slots(env)
    controller_route_identity(env)
    pool_file = env.get('ROLLOUT_ACCOUNT_POOL_FILE', '')
    pool_config = None
    if pool_file:
        from .account_pool import read_config
        pool_config = read_config(pool_file)
        if os.environ.get('ROLLOUT_ATTEMPT', 'auto') != 'auto':
            raise ValueError('Account pool requires ROLLOUT_ATTEMPT=auto for independent retries')
        if env['ROLLOUT_EVALUATION_METHOD'] == 'pi05_only':
            raise ValueError('Disable account pool for pi05_only')
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    repeats = panel.get('collection', {}).get('repeats', 1)
    from .robodojo_server.joint_motion import VERSION as MOTION_VERSION
    print(json.dumps(dict(event='dual_plan', task=env['ROBODOJO_TASK'],
        planned_sampling_episodes=len(cases)*repeats, repeats=repeats,
        panel_sha256=panel['panel_sha256'],
        selection_identity=collection_selection_identity(env, panel),
        motion_executor=MOTION_VERSION if env['ROLLOUT_ENABLE_MOTION_TOOLS'] == '1' else None,
        controller_route=env.get('ROLLOUT_CONTROLLER_ROUTE', 'baseline'),
        cases=[c['case_id'] for c in cases], mode=env['ROLLOUT_RUN_MODE'],
        experiment=env['ROLLOUT_EXPERIMENT_ID'], attempt=env['ROLLOUT_ATTEMPT'],
        auth_profile=env['ROLLOUT_AUTH_PROFILE'],
        account_pool_file=pool_file or None,
        account_pool_profiles=pool_config['profiles'] if pool_config else None,
        model=env['ROLLOUT_MODEL'], reasoning_effort=env['ROLLOUT_REASONING_EFFORT'],
        context_policy=controller_route_identity(env),
        scene_query_enabled=env['ROLLOUT_ENABLE_SCENE_QUERY'] == '1',
        motion_tools_enabled=env['ROLLOUT_ENABLE_MOTION_TOOLS'] == '1',
        pi05_collision_shadow_enabled=env['ROLLOUT_PI05_COLLISION_SHADOW'] == '1',
        simulator=dict(user=env.get('ROLLOUT_SIM_SSH_USER', 'root'), host=env['ROLLOUT_SIM_HOST'], ssh_port=env['ROLLOUT_SIM_SSH_PORT'],
                       gpu=env.get('SIM_GPU', '0'), port=env['ROLLOUT_SIM_REMOTE_PORT'])),
        ensure_ascii=False), flush=True)
    if args.dry_run:
        if len(environments) > 1:
            print(json.dumps(dict(event='parallel_plan', desired_workers=(
                read_desired(env['ROLLOUT_WORKER_CONTROL_FILE']) if env.get('ROLLOUT_WORKER_CONTROL_FILE') else len(environments)), slots=[{k:e[k] for k in
                ('ROLLOUT_WORKER_ID','ROLLOUT_SIM_HOST','ROLLOUT_SIM_SSH_PORT','SIM_GPU',
                 'POLICY_GPU','POLICY_PORT','SIM_PORT','ROLLOUT_SIM_REMOTE_PORT')} for e in environments])), flush=True)
        return
    # Shared resource lock survives exec+account lease; two different accounts
    # cannot accidentally drive the same remote simulator port simultaneously.
    lock_root = Path(env['ROLLOUT_SHARED_ROOT'])/'private_runtime/dual_locks'
    keys = [key for e in environments for key in resource_keys(e)]
    keys += [['local_port', os.uname().nodename, e[k]] for e in environments for k in ('POLICY_PORT','SIM_PORT')]
    resource_fds = acquire_resources(lock_root, keys, wait=args.wait_for_resources)
    # Recheck auto-attempt under the resource lock.
    env, cases = configure(os.environ)
    environments = slots(env)
    env['ROLLOUT_RESOURCE_LOCK_FDS'] = ','.join(map(str, resource_fds))
    print('Checking checkpoint and both native source identities (no model inference)...', flush=True)
    env['ROLLOUT_RESUME_IDENTITY'] = json.dumps(runtime_identity(env), sort_keys=True)
    from .resume_upgrade import prepare as prepare_resume_upgrade
    prepare_resume_upgrade(env)
    if len(environments) > 1:
        checked_hosts = {}
        for e in environments:
            host = identity_endpoint(e)
            if host not in checked_hosts:
                checked_hosts[host] = runtime_identity(e)
            if checked_hosts[host] != json.loads(env['ROLLOUT_RESUME_IDENTITY']):
                raise ValueError('Parallel workers must have identical native source/policy identities')
            e['ROLLOUT_RESUME_IDENTITY'] = env['ROLLOUT_RESUME_IDENTITY']
            e['ROLLOUT_RESOURCE_LOCK_FDS'] = env['ROLLOUT_RESOURCE_LOCK_FDS']
        raise SystemExit(run_parallel(environments, cases))
    if pool_file:
        from .account_pool import run_pool
        raise SystemExit(run_pool(env, cases))
    if env['ROLLOUT_EVALUATION_METHOD'] == 'pi05_only':
        argv = [env['ROBODOJO_PYTHON'], '-u', '-m', 'hybrid_rollout.robodojo.batch']
        os.execvpe(argv[0], argv, env)
    argv = [env['ROBODOJO_PYTHON'], '-m', 'hybrid_rollout.robodojo.codex_backend.profiles',
            '--shared-root', env['ROLLOUT_AUTH_ROOT'], 'lease', '--', env['ROBODOJO_PYTHON'],
            '-u', '-m', 'hybrid_rollout.robodojo.batch']
    os.execvpe(argv[0], argv, env)


if __name__ == '__main__':
    main()

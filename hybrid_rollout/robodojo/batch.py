"""Run one task sequentially, persisting each result before a full process reset."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback

from .artifact_manifest import build_manifest
from .io import write_json
from .evaluation import read_panel, selected_cases, verify_assets, case_identity, archive_path
from .case_ledger import refresh, require_available
from .settings import EFFORT, MODEL, PROVIDER


def utc():
    return datetime.now(timezone.utc).isoformat()


def enable_subreaper():
    # Adopt only this worker's orphan descendants, including Codex's own session.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot isolate rollout descendant cleanup')


def children():
    for _ in range(1000):
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
    path = Path(f'/proc/self/task/{os.getpid()}/children')
    return [int(value) for value in path.read_text().split()]


def reset_descendants(grace=30):
    """Terminate and reap live descendants; never use stored PIDs from another run."""
    signalled = set()
    deadline = time.monotonic() + grace
    while True:
        live = children()
        if not live:
            return {'all_owned_processes_exited': True, 'terminated_descendants': sorted(signalled)}
        sig = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
        for pid in live:
            try:
                os.kill(pid, sig)
                signalled.add(pid)
            except ProcessLookupError:
                pass
        if time.monotonic() > deadline + 10:
            return {'all_owned_processes_exited': False, 'remaining_descendants': live,
                    'terminated_descendants': sorted(signalled)}
        time.sleep(0.1)


def ports_released(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def load_if_exists(path):
    return json.loads(path.read_text()) if path.is_file() else None


def completed_archive(shared, experiment, case, replica, identity, resume_identity):
    """Return a previously verified archive for this exact frozen case.

    A completed *failed* task is still complete for scheduling purposes: it has
    reached the native horizon and has a valid outcome, so rerunning it would
    silently duplicate an evaluation.  Incomplete or unverified archives are
    never skipped.
    """
    base = (Path(shared) / 'results' / experiment / case['task'] /
            case['variant'] / f'eval_seed_{case["eval_seed"]}' /
            f'layout_{case["layout_id"]}' / f'replica_{replica}')
    if not resume_identity:
        raise ValueError('Resume requires a runtime identity; use launch_dual.sh')
    for archive in sorted(list(base.glob('repeat_001/attempt_*')) + list(base.glob('attempt_*')), reverse=True):
        if not archive.is_dir():
            continue
        artifact = load_if_exists(archive / 'artifact_manifest.json') or {}
        result = load_if_exists(archive / 'controller' / 'result.json') or {}
        outcome = load_if_exists(archive / 'sim' / 'evaluation_outcome.json') or {}
        saved_identity = load_if_exists(archive / 'resume_identity.json')
        reset = load_if_exists(archive / 'component_reset.json') or {}
        from .skill_auto_infer.abandon import acknowledged
        abandoned = bool(resume_identity.get('teacher_abandon')) and acknowledged(result, outcome)
        if (artifact.get('status') == 'verified'
                and saved_identity == resume_identity
                and artifact.get('evaluation_method') == resume_identity['evaluation_method']
                and reset.get('all_owned_processes_exited') is True
                and reset.get('ports_released') is True
                and (abandoned or (result.get('complete') is True
                    and outcome.get('complete') is True
                    and outcome.get('valid_for_success_rate') is True
                    and type(outcome.get('native_success')) is bool))
                and result.get('evaluation_case') == identity
                and outcome.get('evaluation_case') == identity):
            return archive
    return None


def run_batch(env, *, episode_command=None):
    env = dict(env)
    from .method import evaluation_method
    method = evaluation_method(env.get('ROLLOUT_EVALUATION_METHOD', 'pi05_plus_gpt'))
    env['ROLLOUT_EVALUATION_METHOD'] = method
    for name in ('ROLLOUT_EXPERIMENT_ID', 'ROBODOJO_TASK', 'ROLLOUT_REPLICA_ID'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', env.get(name, '')):
            raise ValueError(f'Missing or invalid {name}')
    task = env['ROBODOJO_TASK']
    attempt = env.setdefault('ROLLOUT_ATTEMPT', '0')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        raise ValueError('Invalid ROLLOUT_ATTEMPT')
    count = int(env.setdefault('ROLLOUT_COUNT', '1'))
    max_seconds = int(env.setdefault('ROLLOUT_MAX_SECONDS', '10800'))
    if not 1 <= count <= 60 or max_seconds < 0:
        raise ValueError('Require count in 1..6 and non-negative timeout; 0 disables the wall limit')
    if 'ROLLOUT_SEED_BASE' in env:
        raise ValueError('ROLLOUT_SEED_BASE is ambiguous; use a frozen ROLLOUT_EVAL_MANIFEST')
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'], env['ROLLOUT_EVAL_MANIFEST_SHA256'])
    # The validated manifest owns task membership, including new collection
    # tasks. `task` is the base name; standard/random choose `runtime_task`
    # below when the selected case is passed to the episode process.
    if task not in {case['task'] for case in panel['cases']}:
        raise ValueError(f'Base task is not in the frozen panel: {task}; use the base name, not *_random')
    from .collection import runtime_settings
    collection = runtime_settings(env, panel)
    from .regression_eval import runtime_settings as eval_settings
    regression = eval_settings(env, panel)
    repeat_id = None
    if collection:
        repeat_id = int(env['ROLLOUT_COLLECTION_REPEAT_ID'])
        if not 1 <= repeat_id <= panel['collection']['repeats'] or count != 1:
            raise ValueError('Collection requires one case and repeat_id within panel repeats')
        job_index = int(env['ROLLOUT_COLLECTION_JOB_INDEX'])
        from .collection import sampling_job_index
        if job_index != sampling_job_index(panel, env.get('ROLLOUT_CASE_ID'), repeat_id):
            raise ValueError('Collection job/case/repeat identity mismatch')
    explicit_case = env.get('ROLLOUT_CASE_ID')
    if method == 'gpt_only' and episode_command is None:
        from .phase_gate import verify_certificate
        verify_certificate(env.get('ROLLOUT_PHASE1_CERTIFICATE'), panel, explicit_case)
    # Production schedulers normally reserve exactly one explicit case per
    # worker.  A deliberately opt-in sequential campaign may consume the
    # frozen replica order in one worker; it still runs one fresh process tree
    # and archive per case.
    sequential_campaign = env.get('ROLLOUT_ALLOW_SEQUENTIAL_BATCH') == '1'
    if episode_command is None and (count != 1 or not explicit_case) and not sequential_campaign:
        raise ValueError('New production workers require ROLLOUT_COUNT=1 and an explicit ROLLOUT_CASE_ID')
    cases = selected_cases(panel, task, env['ROLLOUT_REPLICA_ID'], count, explicit_case)
    if env.get('ROLLOUT_CASE_IDS'):
        if not sequential_campaign or explicit_case:
            raise ValueError('CASE_IDS requires sequential mode without CASE_ID')
        ids = env['ROLLOUT_CASE_IDS'].split(',')
        from .evaluation import replica_cases
        candidates = replica_cases(panel, task, env['ROLLOUT_REPLICA_ID'])
        if len(ids) != count or len(set(ids)) != count or not set(ids) <= {c['case_id'] for c in candidates}:
            raise ValueError('Invalid CASE_IDS selection')
        cases = [next(c for c in candidates if c['case_id'] == name) for name in ids]
    verify_source = env.get('ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY') != '1'
    skip_completed = env.get('ROLLOUT_SKIP_COMPLETED_CASES') == '1'
    resume_identity = json.loads(env.get('ROLLOUT_RESUME_IDENTITY', 'null'))
    if skip_completed and not resume_identity:
        raise ValueError('Resume identity missing; use launch_dual.sh, not legacy exports')
    if episode_command is None:
        verify_assets(panel, env['ROBODOJO_SOURCE'], cases, verify_source=verify_source)
    shared = Path(env.setdefault('ROLLOUT_SHARED_ROOT', env.get('RUNTIME_ROOT',
        '/mnt/rollout/robodojo_mixed_control'))).resolve()
    if episode_command is None and (not shared.is_relative_to('/mnt') or shared == Path('/mnt')):
        raise ValueError('Shared output root must be below /mnt')
    experiment, replica = env['ROLLOUT_EXPERIMENT_ID'], env['ROLLOUT_REPLICA_ID']
    batch_root = shared/'results'/experiment/'_replicas'/f'replica_{replica}'/f'attempt_{attempt}'
    rerun_of = json.loads(env.get('ROLLOUT_RERUN_REFERENCE', 'null'))
    if explicit_case and not sequential_campaign:
        require_available(shared, panel, explicit_case, batch_root/'batch.json', rerun_of=rerun_of, method=method)
    if skip_completed:
        for old in batch_root.parent.glob('attempt_*/batch.json'):
            saved = load_if_exists(old)
            if saved.get('resume_identity') != resume_identity:
                raise ValueError(f'Resume configuration differs or is missing in {old}; use a new experiment')
    batch_root.parent.mkdir(parents=True, exist_ok=True)
    batch_root.mkdir()  # Atomic reservation: repeated scheduler launches cannot overwrite.
    write_json(batch_root/'evaluation_manifest.json', panel)
    if resume_identity and episode_command is None:
        from .deployment import snapshot
        frozen = batch_root/'source_snapshot'
        if snapshot(env['CODE_ROOT'], frozen) != resume_identity['source_sha256']:
            raise ValueError('Source changed during launch preparation; start a new attempt')
        env['CODE_ROOT'] = str(frozen)
        env['ROLLOUT_EPISODE_ENTRYPOINT'] = str(frozen/'hybrid_rollout/robodojo/dual_machine_episode_entrypoint.sh')
    ports = [int(env.get('POLICY_PORT', 18830)), int(env.get('SIM_PORT', 19113))]
    if episode_command is not None:
        command = episode_command
    else:
        # Keep the native single-host launcher as the default.  The dual-host
        # launcher is opt-in so existing RoboDojo jobs remain unchanged.
        launcher = env.get('ROLLOUT_EPISODE_ENTRYPOINT')
        command = ['bash', launcher] if launcher else [
            'bash', str(Path(__file__).with_name('episode_entrypoint.sh'))]
    enable_subreaper()
    from .prompt_context import CONTEXT_VERSION
    manifest = dict(schema='hybrid_rollout.robodojo.batch.v2', state='running',
        evaluation_method=method,
        resume_identity=resume_identity,
        auth_profile=env.get('ROLLOUT_AUTH_PROFILE'),
        context_version=CONTEXT_VERSION,
        rerun_of=rerun_of,
        started_utc=utc(), task=task, replica=replica, attempt=attempt, planned_rollouts=count,
        panel_id=panel['panel_id'], panel_sha256=panel['panel_sha256'],
        eval_seed=panel['eval_seed'], model=MODEL, effort=EFFORT, provider=PROVIDER, episodes=[])
    write_json(batch_root/'batch.json', manifest)
    def refresh_progress():
        # The official ledger assumes one completion per case. Collection has
        # five intentional samples; its authoritative index is pool_run.json.
        # Regression uses its own experiment-scoped report: the old ledger
        # treats baseline + trained policy results as duplicate completions.
        if not collection and not regression:
            refresh(shared, env['ROLLOUT_EVAL_MANIFEST'], method=method)
    refresh_progress()
    active = None
    exit_code = 0
    def interrupted(signum, frame):
        raise InterruptedError(f'Batch received signal {signum}')
    previous_signal_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, case in enumerate(cases):
            if episode_command is None:
                verify_assets(panel, env['ROBODOJO_SOURCE'], [case], verify_source=verify_source)
            sample_attempt = env.get('ROLLOUT_SAMPLE_ATTEMPT', attempt)
            if 'ROLLOUT_SAMPLE_ATTEMPT' in env and not str(sample_attempt).isdigit():
                raise ValueError('ROLLOUT_SAMPLE_ATTEMPT must be a nonnegative integer')
            if collection and 'ROLLOUT_SAMPLE_ATTEMPT' not in env:
                from .evaluation import reserve_sample_attempt
                sample_attempt = reserve_sample_attempt(shared, experiment, case, replica, repeat_id)
            archive = archive_path(shared, experiment, case, replica, sample_attempt, repeat_id or 1)
            identity = case_identity(panel, case)
            case_file = batch_root/f'case_{index:03d}.json'
            write_json(case_file, dict(identity=identity, case=case))
            previous = completed_archive(shared, experiment, case, replica, identity, resume_identity) if skip_completed else None
            if previous is not None:
                row = dict(index=index, **identity, archive=str(previous),
                           state='skipped_completed', started_utc=utc(),
                           finished_utc=utc(), skipped_reason='previous_verified_archive',
                           result=load_if_exists(previous/'controller'/'result.json'),
                           evaluation_outcome=load_if_exists(previous/'sim'/'evaluation_outcome.json'),
                           artifact_status='verified')
                manifest['episodes'].append(row)
                write_json(batch_root/'batch.json', manifest)
                refresh_progress()
                print(json.dumps(dict(event='rollout_skip', index=index,
                                      case_id=case['case_id'], archive=str(previous),
                                      reason='previous_verified_archive')), flush=True)
                continue
            row = dict(index=index, **identity, archive=str(archive), state='starting', started_utc=utc(),
                       archive_layout='repeat_attempt_v1', sample_attempt=sample_attempt, batch_attempt=attempt,
                       auth_profile=env.get('ROLLOUT_AUTH_PROFILE'),
                       account_fingerprint=env.get('ROLLOUT_ACCOUNT_FINGERPRINT'))
            if collection:
                row.update(repeat_id=repeat_id,
                           collection_job_index=int(env['ROLLOUT_COLLECTION_JOB_INDEX']))
            manifest['episodes'].append(row)
            write_json(batch_root/'batch.json', manifest)
            refresh_progress()
            if not ports_released(ports):
                raise RuntimeError('A rollout port is already owned; refusing to start another component')
            child_env = dict(env, ROBODOJO_TASK=case['runtime_task'],
                             ROLLOUT_SAMPLE_ATTEMPT=str(sample_attempt),
                             ROLLOUT_BATCH_RECEIPT=str(batch_root/'batch.json'),
                             ROLLOUT_BASE_TASK=case['task'], ROLLOUT_VARIANT=case['variant'],
                             ROLLOUT_EVAL_SEED=str(case['eval_seed']),
                             ROLLOUT_LAYOUT_ID=str(case['layout_id']), ROLLOUT_INDEX=str(index),
                             ROLLOUT_CASE_FILE=str(case_file), ROLLOUT_ARCHIVE=str(archive))
            if panel.get('queue_mode') == 'all_policy_seeds_v1':
                child_env['ROLLOUT_POLICY_RNG_SEED'] = str(case['policy_rng_seed'])
            print(json.dumps(dict(event='rollout_start', **row)), flush=True)
            with (batch_root/f'rollout_{index:03d}.log').open('x') as log:
                episode_started_mono = time.monotonic()
                active = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                try:
                    row['exit_code'] = active.wait(timeout=max_seconds or None)
                except subprocess.TimeoutExpired:
                    row['exit_code'] = 124
                    row['error'] = 'Per-rollout wall time limit reached'
                    row['failure_kind'] = 'wall_time_limit'
                finally:
                    if active.poll() is None:
                        active.terminate()
                        try:
                            active.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            active.kill()
                            active.wait()
                    active = None
            reset = reset_descendants()
            reset['ports_released'] = ports_released(ports)
            reset['finished_utc'] = utc()
            row['reset'] = reset
            write_json(batch_root/f'reset_{index:03d}.json', reset)
            row['result'] = load_if_exists(archive/'controller/result.json')
            row['evaluation_outcome'] = load_if_exists(archive/'sim/evaluation_outcome.json')
            from .skill_auto_infer.abandon import acknowledged
            if (env.get('ROLLOUT_ENABLE_TEACHER_ABANDON') == '1'
                    and acknowledged(row['result'] or {}, row['evaluation_outcome'] or {})):
                row.update(failure_kind='teacher_abandon', task_success=False,
                           collection_complete=True, native_complete=False)
            row['token_usage'] = load_if_exists(archive/'controller/token_usage.json')
            if env.get('ROLLOUT_ACCOUNT_POOL_FILE') and row['exit_code'] != 0 and not row.get('failure_kind'):
                from .account_pool import failure_kind
                failure = load_if_exists(archive/'controller/failure.json') or {}
                kind = failure_kind(failure.get('error'))
                if kind:
                    row['failure_kind'] = kind
                    if kind == 'simulator_connection_failed':
                        row['connection_episode_retry'] = int(env.get('ROLLOUT_CONNECTION_EPISODE_RETRY', '0'))
                    if kind == 'model_recovery_exhausted':
                        row['model_episode_retry'] = int(env.get('ROLLOUT_MODEL_EPISODE_RETRY', '0'))
            # Hash only after every component has exited, including orphaned native tools.
            if archive.is_dir():
                if env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
                    write_json(archive/'account_attempt.json', dict(
                        case_id=case['case_id'], attempt=sample_attempt, batch_attempt=attempt,
                        repeat_id=repeat_id,
                        auth_profile=row['auth_profile'], account_fingerprint=row['account_fingerprint'],
                        exit_code=row['exit_code'], failure_kind=row.get('failure_kind'),
                        started_utc=row['started_utc'], finished_utc=utc()))
                from .timing import summarize
                summarize(archive/'timing', lifecycle=dict(case_id=case['case_id'],
                    task=case['task'], exit_code=row['exit_code'], error=row.get('error'),
                    launch_through_cleanup_s=time.monotonic()-episode_started_mono,
                    started_utc=row['started_utc'], stopped_utc=utc(),
                    note='Excludes the following archive hashing/ledger update.'))
                write_json(archive/'component_reset.json', reset)
                artifact = build_manifest(archive)
                write_json(archive/'artifact_manifest.json', artifact)
                row['artifact_status'] = artifact['status']
                # Native close can finish publishing its outcome after the early
                # result read. Hashing is complete and all owned processes exited.
                if row.get('evaluation_outcome') is None:
                    row['evaluation_outcome'] = load_if_exists(archive/'sim/evaluation_outcome.json')
            okay = (row['exit_code'] == 0 and row.get('artifact_status') == 'verified'
                    and reset['all_owned_processes_exited'] and reset['ports_released'])
            row.update(state='completed' if okay else 'failed', finished_utc=utc())
            write_json(batch_root/'batch.json', manifest)
            refresh_progress()
            print(json.dumps(dict(event='rollout_done', index=index, state=row['state'],
                                  archive=str(archive), reset=reset,
                                  error=row.get('error'))), flush=True)
            if not okay:
                exit_code = row['exit_code'] or 5
                # A wall timeout is a failed case, not permission to reuse an
                # unclean simulator. Preserve native outcome/incomplete evidence.
                remote_dir = archive/'dual_machine'
                receipt = remote_dir/'remote_cleanup_status.txt'
                remote_clean = (not remote_dir.exists() or
                                (receipt.is_file() and receipt.read_text().strip() == '0'))
                if (row.get('failure_kind') == 'wall_time_limit'
                        and row.get('artifact_status') == 'verified'
                        and reset['all_owned_processes_exited'] and reset['ports_released']
                        and remote_clean):
                    print(json.dumps(dict(event='rollout_timeout_continue', index=index,
                                          archive=str(archive), error=row['error'])), flush=True)
                    continue
                break  # Do not spend more tokens or silently retry after infrastructure failure.
    except BaseException as error:
        exit_code = 130 if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 2
        manifest['error'] = traceback.format_exc()
    finally:
        for sig in previous_signal_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=120)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()
        manifest['final_reset'] = reset_descendants()
        manifest['final_reset']['ports_released'] = ports_released(ports)
        if not all(manifest['final_reset'][k] for k in ('all_owned_processes_exited', 'ports_released')):
            exit_code = exit_code or 5
        manifest.update(state='completed' if exit_code == 0 else 'failed',
                        exit_code=exit_code, finished_utc=utc())
        write_json(batch_root/'batch.json', manifest)
        refresh_progress()
        for sig, handler in previous_signal_handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(event='batch_done', state=manifest['state'], exit_code=exit_code,
                          batch_path=str(batch_root/'batch.json'),
                          completed=sum(r['state'] == 'completed' for r in manifest['episodes']),
                          failed=sum(r['state'] == 'failed' for r in manifest['episodes']),
                          skipped=sum(r['state'] == 'skipped_completed' for r in manifest['episodes']))),
          flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(run_batch(os.environ))

"""Opt-in serial episode scheduling across explicitly authorized account profiles.

No mid-episode handoff, credential copying, model fallback or credit redemption.
Locks are authoritative for occupancy; persisted quota observations always expire.
"""
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from .codex_backend.profiles import account_home, account_lock, validate_credential, CHATGPT_PROFILES
from .io import write_json
from .quota_rpc import account_rpc, codex_limits


class WorkerRetired(InterruptedError):
    """Scheduler requested a boundary stop, not an episode failure."""


def read_config(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Account pool config must be an object')
    groups = value.get('accounts')
    if groups is not None:
        if 'profiles' in value or not isinstance(groups, list):
            raise ValueError('Use either accounts or profiles, not both')
        labels = []
        for group in groups:
            if (not isinstance(group, dict) or not isinstance(group.get('name'), str)
                    or not group['name'] or not isinstance(group.get('profiles'), list)
                    or not group['profiles'] or type(group.get('max_concurrent')) is not int
                    or not 1 <= group['max_concurrent'] <= min(8, len(group['profiles']))
                    or group.get('priority', 'primary') not in ('primary', 'backup')):
                raise ValueError('Each account needs name, profiles, max_concurrent (1..8), priority (primary/backup)')
            labels.append(group['name'])
        if len(set(labels)) != len(labels):
            raise ValueError('Duplicate account names')
        groups = sorted(groups, key=lambda g: g.get('priority', 'primary') == 'backup')
        value['accounts'] = groups
        value['profiles'] = [n for g in groups for n in g['profiles']]
    names = value.get('profiles')
    if not isinstance(names, list) or any(not isinstance(n, str) or n not in CHATGPT_PROFILES for n in names):
        raise ValueError('profiles must be a list of managed ChatGPT profile names')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate profiles')
    for key, default in (('check_interval_seconds', 60), ('quota_recheck_seconds', 600)):
        value.setdefault(key, default)
        if type(value[key]) is not int or not 10 <= value[key] <= 3600:
            raise ValueError(f'{key} must be an integer in 10..3600')
    return value


def failure_kind(error):
    """Classify terminal controller failures only, never arbitrary tool/log text."""
    # Only the final exception line: preceding trace/source lines can contain
    # words such as quota without being the actual failure.
    text = str(error).strip().splitlines()[-1].lower() if error else ''
    # Only unwrap the observed wrapper and its immediate explicit Python cause.
    # Never scan arbitrary source lines / model text for transport keywords.
    if text == 'runtimeerror: feedforward inference failed; abort without replay or candidate resampling':
        blocks = str(error).strip().split('\nThe above exception was the direct cause of the following exception:\n')
        if len(blocks) >= 2:
            cause = blocks[-2].strip().splitlines()[-1]
            if cause.startswith('hybrid_rollout.robodojo.robodojo_server.protocol.SimulatorConnectionError: Simulator RPC transport failure during '):
                return 'simulator_connection_failed'
    prefix = 'RuntimeError: Codex error: '
    last = str(error).strip().splitlines()[-1] if error else ''
    if last.startswith(prefix):
        import ast
        try:
            event = ast.literal_eval(last[len(prefix):])
        except (SyntaxError, ValueError):
            event = None
        if isinstance(event, dict):
            info = event.get('error')
            if (isinstance(info, dict) and info.get('codexErrorInfo') == 'other'
                    and event.get('willRetry') is False):
                # Exact observed remote-compaction rejection only. Do not retry
                # arbitrary HTTP 400/schema errors or text quoted by the model.
                message = info.get('message')
                compact_prefix = 'Error running remote compact task: '
                if isinstance(message, str) and message.startswith(compact_prefix):
                    try:
                        detail = json.loads(message[len(compact_prefix):])
                    except ValueError:
                        detail = None
                    if detail == {'detail': 'Bad Request'}:
                        return 'codex_compaction_failed'
            if (isinstance(info, dict) and info.get('message') == 'workspace routing discovery failed'
                    and info.get('codexErrorInfo') == 'other' and event.get('willRetry') is False):
                return 'codex_transport_failed'
    if text in ('timeouterror: timed out waiting for a codex turn event',
                'timeouterror: codex made no completed service or native tool call within the configured timeout'):
        return 'codex_transport_failed'
    if text == ('hybrid_rollout.robodojo.skill.network_recovery.codextransportrecoveryerror: '
                'disconnected ephemeral codex turn timed out; terminal state unavailable'):
        return 'codex_transport_failed'
    # Compatibility with immutable pre-fix episode snapshots. Require the exact
    # terminal RPC rejection AND its timeout traceback, not a keyword in GPT text.
    if (text == "runtimeerror: codex rpc thread/read failed: {'code': -32600, 'message': 'ephemeral threads do not support includeturns'}"
            and 'TimeoutError: Timed out waiting for a Codex turn event' in str(error)):
        return 'codex_transport_failed'
    if (text == 'eoferror: rpc connection closed' or text.startswith(
            'hybrid_rollout.robodojo.robodojo_server.protocol.simulatorconnectionerror: '
            'simulator rpc transport failure during ')):
        return 'simulator_connection_failed'
    if text == ('hybrid_rollout.robodojo.skill_auto_infer.model_recovery.modelrecoveryexhausted: '
                'ten same-thread model recovery attempts exhausted'):
        return 'model_recovery_exhausted'
    if re.search(r'usage.?limit.?exceeded|usage[_ ]limit|insufficient[_ ]quota|'
                 r'quota.{0,25}(exhaust|exceed)|credit.{0,20}(exhaust|deplet)|'
                 r"(you.ve|you have) reached your.*limit", text):
        return 'quota_interrupted'
    if re.search(r'authentication|unauthorized|invalid[_ ]api[_ ]key|'
                 r'token.{0,12}expired|subscription.{0,20}(expired|inactive)', text):
        return 'auth_interrupted'
    return None


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def archived_failure_kind(row):
    """Reclassify immutable legacy receipts without editing episode artifacts."""
    archive = Path(row.get('archive') or '/nonexistent')
    failure = load_json(archive/'controller/failure.json')
    reason = row.get('failure_kind') or failure_kind(failure.get('error'))
    if reason == 'model_recovery_exhausted':
        # Exhaustion can also be a semantic/model error. Upgrade only with the
        # final structured terminal event proving a connection failure at this step.
        log = archive/'logs/controller.log'
        last = None
        if log.is_file():
            with log.open(errors='replace') as stream:
                for line in stream:
                    try: event = json.loads(line)
                    except ValueError: continue
                    if isinstance(event, dict) and event.get('event') == 'codex_model_wait_terminal':
                        last = event
        from .skill.network_recovery import is_network_error
        if (last and failure.get('step_id') is not None
                and last.get('step_id') == failure['step_id']
                and isinstance(last.get('error'), dict) and is_network_error(last['error'])):
            return 'codex_transport_failed'
    return reason


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)


def fingerprint(path):
    value = json.loads(path.read_text()).get('tokens', {}).get('account_id')
    if not isinstance(value, str) or not value:
        raise ValueError('Account identity unavailable; log in again')
    return hashlib.sha256(value.encode()).hexdigest()


def next_check(limits, now, interval):
    # Periodic rechecks also discover purchased/renewed quota before a scheduled
    # reset. A reset time is only a hint, never permission to use the account.
    times = [w.get('resetsAt') for w in limits.get('windows', [])
             if w.get('usedPercent', 0) >= 100]
    future = [t + 15 for t in times if isinstance(t, (int, float)) and t + 15 > now]
    return min(now + interval, max(future)) if future else now + interval


class AccountPool:
    def __init__(self, env, *, probe=None, clock=time.time, sleep=time.sleep):
        self.env = env
        self.config_path = Path(env['ROLLOUT_ACCOUNT_POOL_FILE']).resolve()
        self.root = Path(env['ROLLOUT_AUTH_ROOT'])/'private/account_pool'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.probe = probe or self.read_limits
        self.clock, self.sleep = clock, sleep

    def read_limits(self, name):
        with account_rpc(name, self.env['ROLLOUT_AUTH_ROOT'], self.env['CODEX_BIN'],
                         self.env.get('ROLLOUT_HTTPS_PROXY_FILE')) as rpc:
            result = codex_limits(rpc.request('account/rateLimits/read'))
        if result is None:
            raise ValueError('Codex quota unavailable')
        return result

    def state_path(self, identity):
        return self.root/f'{identity}.json'

    @contextmanager
    def state_lock(self, identity):
        """Serialize quota probes and cooldown writes across concurrent slots."""
        fd = os.open(self.root/f'{identity}.state.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def concurrency(self, config, name, identity):
        groups = config.get('accounts')
        if groups is None:
            return 1  # Preserve legacy physical-account deduplication.
        selected = next(g for g in groups if name in g['profiles'])
        sessions = set()
        for group in groups:
            for alias in group['profiles']:
                try:
                    credential = validate_credential(alias, self.env['ROLLOUT_AUTH_ROOT'])
                    other = fingerprint(credential)
                except (ValueError, OSError, KeyError):
                    continue  # Missing/broken logins are unavailable, not capacity.
                if (group is selected) != (other == identity):
                    if group is selected or other == identity:
                        raise ValueError('account_group_identity_mismatch')
                if group is selected and selected['max_concurrent'] > 1:
                    token = json.loads(credential.read_text()).get('tokens', {}).get('refresh_token')
                    if not token or token in sessions:
                        raise ValueError('independent_logins_required')
                    sessions.add(token)
        return selected['max_concurrent']

    def block(self, identity, reason, revision=None):
        # Caller still holds the account lock. Do not persist raw error text.
        now = self.clock()
        try:
            interval = read_config(self.config_path)['quota_recheck_seconds']
        except (ValueError, OSError):
            interval = 600
        with self.state_lock(identity):
            write_json(self.state_path(identity), dict(status=reason, observed_at=now,
                       next_check_at=now + interval, credential_revision=revision))

    @contextmanager
    def acquire(self, job, on_wait=None):
        started = self.clock()
        while True:
            if getattr(self, 'cancelled', lambda: False)():
                raise WorkerRetired('Parallel dispatch stopped; no new account lease')
            reasons = {}
            interval = 60
            try:
                config = read_config(self.config_path)
                interval = config['check_interval_seconds']
                names = config['profiles']
                if not names:
                    reasons['pool'] = 'empty_profile_list'
            except (ValueError, OSError):
                names = []
                reasons['config'] = 'invalid_or_missing_config; edit the pool file'
            for name in names:
                with ExitStack() as locks:
                    try:
                        profile_fd = locks.enter_context(account_lock(name, self.env['ROLLOUT_AUTH_ROOT']))
                    except ValueError as error:
                        reasons[name] = 'busy' if 'is in use' in str(error) else 'login_directory_invalid'
                        continue
                    try:
                        credential = validate_credential(name, self.env['ROLLOUT_AUTH_ROOT'])
                        identity = fingerprint(credential)
                        revision = hashlib.sha256(credential.read_bytes()).hexdigest()
                    except (ValueError, OSError, KeyError):
                        reasons[name] = 'login_required'
                        continue
                    try:
                        capacity = self.concurrency(config, name, identity)
                    except ValueError as error:
                        reasons[name] = ('independent_logins_required' if str(error) == 'independent_logins_required'
                                         else 'account_group_identity_mismatch')
                        continue
                    fd = os.open(self.root/f'{identity}.lock', os.O_CREAT | os.O_RDWR, 0o600)
                    locks.callback(os.close, fd)
                    try:
                        # Shared guard is incompatible with old exclusive leases.
                        fcntl.flock(fd, (fcntl.LOCK_SH if capacity > 1 else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                    except BlockingIOError:
                        reasons[name] = 'same_account_busy'
                        continue
                    slot_fd = None
                    for slot in range(capacity):
                        candidate = os.open(self.root/f'{identity}.slot_{slot}.lock', os.O_CREAT | os.O_RDWR, 0o600)
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            os.close(candidate)
                            continue
                        slot_fd = candidate
                        locks.callback(os.close, slot_fd)
                        break
                    if slot_fd is None:
                        reasons[name] = 'account_concurrency_limit'
                        continue
                    with self.state_lock(identity):
                        now = self.clock()
                        try:
                            state = load_json(self.state_path(identity))
                        except ValueError:
                            state = {}  # Corrupt cache cannot permanently disable an account.
                        if state.get('next_check_at', 0) > now:
                            # A repaired login can bypass an authentication cooldown,
                            # but a different login session cannot bypass account quota.
                            changed_login = (state.get('status') == 'auth_interrupted'
                                             and state.get('credential_revision') != revision)
                            if not changed_login:
                                reasons[name] = state.get('status', 'cooldown')
                                continue
                        try:
                            emit('account_pool_check', auth_profile=name, job=job)
                            limits = self.probe(name)
                            if limits['remaining_percent'] <= 0 or limits.get('reached_type'):
                                state = dict(status='quota_exhausted', observed_at=now,
                                    next_check_at=next_check(limits, now, config['quota_recheck_seconds']))
                                write_json(self.state_path(identity), state)
                                reasons[name] = 'quota_exhausted'
                                continue
                        except Exception as error:
                            # Do not print exceptions from authentication RPCs (credentials).
                            state = dict(status='quota_check_failed', observed_at=now,
                                         next_check_at=now + interval, error_type=type(error).__name__)
                            write_json(self.state_path(identity), state)
                            reasons[name] = 'quota_check_failed'
                            continue
                        write_json(self.state_path(identity), dict(status='available', observed_at=now,
                                   next_check_at=0, credential_revision=revision))
                    if getattr(self, 'cancelled', lambda: False)():
                        raise WorkerRetired('Worker draining; do not start another episode')
                    emit('account_pool_selected', job=job, auth_profile=name,
                         account_fingerprint=identity, waited_seconds=round(now-started))
                    owner = self.root/(f'{identity}.slot_{slot}.owner.json' if capacity > 1 else f'{identity}.owner.json')
                    write_json(owner, dict(auth_profile=name, pid=os.getpid(), host=os.uname().nodename,
                               job=job, started_at=now, slot=slot, max_concurrent=capacity, note='Display only; OS lock is authoritative'))
                    try:
                        yield dict(auth_profile=name, account_fingerprint=identity,
                                   credential_revision=revision, slot=slot, fds=(profile_fd, fd, slot_fd))
                    finally:
                        owner.unlink(missing_ok=True)
                    return
            elapsed = round(self.clock() - started)
            message = (f'\n!!!!!!!! ACCOUNT POOL BLOCKED / 账号全部不可用 !!!!!!!!\n'
                       f'本 worker 任务未运行，正在等待账号（其他 worker 不受影响） | job={job} | 已等待 {elapsed} 秒\n'
                       f'下次检查 {interval} 秒后；添加/重新登录账号后将自动重读：{self.config_path}\n'
                       f'原因：{json.dumps(reasons, ensure_ascii=False)}\n'
                       '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(message, flush=True)
            emit('account_pool_wait', level='ERROR', job=job, waited_seconds=elapsed,
                 next_check_seconds=interval, reasons=reasons)
            if on_wait:
                on_wait(dict(waited_seconds=elapsed, next_check_seconds=interval, reasons=reasons))
            # Short sleeps make Ctrl-C/SIGTERM responsive. No busy spinning.
            remaining = interval
            while remaining:
                delay = min(remaining, 5)
                self.sleep(delay)
                remaining -= delay


def clean_result(batch, row):
    for receipt in (batch.get('final_reset', {}), row.get('reset', {})):
        if not all(receipt.get(k) is True for k in ('all_owned_processes_exited', 'ports_released')):
            return False
    remote = Path(row['archive'])/'dual_machine/remote_cleanup_status.txt'
    return remote.is_file() and remote.read_text().strip() == '0'


def run_child(env, fds):
    from .auth_profile_compat import prepare
    env = prepare(env)
    fds = tuple(fds) + tuple(int(x) for x in env.get('ROLLOUT_RESOURCE_LOCK_FDS', '').split(',') if x)
    child = subprocess.Popen([env['ROBODOJO_PYTHON'], '-u', '-m', 'hybrid_rollout.robodojo.batch'],
                             env=env, pass_fds=fds, start_new_session=True)
    try:
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()  # batch performs owned-tree and remote cleanup.
            try:
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise RuntimeError('Batch cleanup timed out; inspect remote processes before restarting')


def completion_receipt(row, runtime):
    """Read a late-written native result without mutating an old batch receipt."""
    if row.get('evaluation_outcome') is not None or row.get('state') != 'completed':
        return row
    archive = Path(row['archive'])
    artifact = load_json(archive/'artifact_manifest.json')
    if row.get('artifact_status') != 'verified' or artifact.get('status') != 'verified':
        return row
    files = {f['path']: f for f in artifact.get('all_immutable_artifacts', [])}
    values = {}
    for name in ('sim/evaluation_outcome.json', 'controller/result.json', 'resume_identity.json',
                 'collection_job.json'):
        p = archive/name
        if name not in files or not p.is_file():
            return row
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != files[name]['sha256']:
            return row
        values[name] = json.loads(raw)
    outcome = values['sim/evaluation_outcome.json']
    result = values['controller/result.json']
    job = values['collection_job.json']
    if (values['resume_identity.json'] != runtime
            or (job.get('case_id'), job.get('repeat_id')) != (row['case_id'], row.get('repeat_id'))
            or result.get('evaluation_case') != outcome.get('evaluation_case')
            or (outcome.get('evaluation_case') or {}).get('case_id') != row['case_id']
            or result.get('complete') is not True or outcome.get('complete') is not True
            or result.get('episode_id') != artifact.get('episode_id')
            or result.get('step_id') != outcome.get('native_control_steps')
            or outcome.get('valid_for_success_rate') is not True
            or type(outcome.get('native_success')) is not bool):
        return row
    return dict(row, result=result, evaluation_outcome=outcome,
                completion_evidence='hashed_archive_native_outcome')


def collection_completed(replica_root, runtime, case_id, repeat_id):
    """A quota interruption is not a sample; a cleaned timeout is a sample."""
    from .resume_upgrade import accepted_identities
    accepted = accepted_identities(replica_root.parent.parent, runtime)
    for path in sorted(replica_root.glob('attempt_*/batch.json')):
        batch = load_json(path)
        if batch.get('resume_identity') not in accepted:
            continue
        for row in batch.get('episodes', []):
            if (row.get('case_id'), row.get('repeat_id')) != (case_id, repeat_id):
                continue
            if not clean_result(batch, row):
                continue
            row = completion_receipt(row, batch.get('resume_identity'))
            if row.get('failure_kind') == 'wall_time_limit':
                return row
            if (row.get('failure_kind') == 'model_recovery_exhausted'
                    and row.get('model_episode_retry') == 1):
                return row  # Terminal infrastructure failure, not a valid training sample.
            from .skill_auto_infer.abandon import acknowledged
            if (runtime.get('teacher_abandon') and row.get('state') == 'completed'
                    and row.get('artifact_status') == 'verified'
                    and acknowledged(row.get('result') or {}, row.get('evaluation_outcome') or {})):
                return row
            if (row.get('state') == 'completed' and row.get('artifact_status') == 'verified'
                    and (row.get('result') or {}).get('complete')
                    and (row.get('evaluation_outcome') or {}).get('complete')):
                return row
    return None


def model_retry_used(replica_root, runtime, case_id, repeat_id):
    """Persist the retry budget across worker retirement/restarts, not account changes."""
    for path in replica_root.glob('attempt_*/batch.json'):
        batch = load_json(path)
        if batch.get('resume_identity') != runtime:
            continue
        for row in batch.get('episodes', []):
            if ((row.get('case_id'), row.get('repeat_id')) == (case_id, repeat_id)
                    and row.get('failure_kind') == 'model_recovery_exhausted'
                    and clean_result(batch, row)):
                return True
    return False


def save_sample_summary(env, case, job):
    from .evaluation import archive_path
    sample = archive_path(env['ROLLOUT_SHARED_ROOT'], env['ROLLOUT_EXPERIMENT_ID'],
                          case, env['ROLLOUT_REPLICA_ID'], 0, job['repeat_id']).parent
    sample.mkdir(parents=True, exist_ok=True)
    previous = load_json(sample/'sample.json')
    attempts = {r['batch_attempt']: r for r in previous.get('attempts', [])}
    attempts.update({r['batch_attempt']: r for r in job['attempts']})
    write_json(sample/'sample.json', dict(schema='robodojo.sample.v1',
        case_id=job['case_id'], repeat_id=job['repeat_id'], state=job['state'],
        pending_reason=job.get('pending_reason'),
        attempts=list(attempts.values()), archive=job.get('archive') or
        (job['attempts'][-1].get('archive') if job['attempts'] else previous.get('archive'))))


def run_pool(env, cases, *, runner=run_child, pool=None, job_source=None, allocate_attempt=None,
             raise_on_retire=False):
    """One existing batch per case attempt; keep native archives/validation intact."""
    from .batch import completed_archive
    from .deployment import snapshot
    from .evaluation import case_identity, read_panel

    env = dict(env)
    if env['ROLLOUT_EVALUATION_METHOD'] == 'pi05_only':
        raise ValueError('Account pool is for teacher policies only')
    pool = pool or AccountPool(env)
    base = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/env['ROLLOUT_EXPERIMENT_ID']
    replica_root = base/'_replicas'/f'replica_{env["ROLLOUT_REPLICA_ID"]}'
    runtime = json.loads(env['ROLLOUT_RESUME_IDENTITY'])
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    from .collection import SCHEMA
    collection = panel.get('schema') == SCHEMA
    repeats = panel['collection']['repeats'] if collection else 1
    jobs = [(repeat if collection else None, case)
            for repeat in range(1, repeats + 1) for case in cases]
    from . import connection_recovery
    # Unified eval queues carry the same shared monitor as collection queues.
    # Previously it was ignored unless the panel used the collection schema.
    recovery = getattr(pool, 'connection_recovery', None)
    worker_index = int(env.get('ROLLOUT_WORKER_ID', 'worker_0').rsplit('_', 1)[-1])
    if collection:
        if recovery is None:
            import multiprocessing as mp
            ctx = mp.get_context('fork')
            recovery = pool.connection_recovery = connection_recovery.ConnectionRecovery(
                ctx, ctx.Event(), len(jobs))
    def cancelled():
        return ((recovery is not None and recovery.stop.is_set())
                or getattr(pool, 'cancelled', lambda: False)())
    if env['ROLLOUT_RUN_MODE'] == 'resume':
        from .resume_upgrade import accepted_identities
        accepted = accepted_identities(base, runtime)
        for old in replica_root.glob('attempt_*/batch.json'):
            if load_json(old).get('resume_identity') not in accepted:
                raise ValueError('Resume identity differs; use a new experiment')
        for old in (base/'_account_pool').glob('run_*/pool_run.json'):
            if load_json(old).get('resume_identity') not in accepted:
                raise ValueError('Account pool resume identity differs; use a new experiment')
    directory = base/'_account_pool'/f'run_{uuid.uuid4().hex}'
    directory.mkdir(parents=True)
    # Freeze once so edits during a long wait cannot silently change later jobs.
    frozen = directory/'source_snapshot'
    if snapshot(env['CODE_ROOT'], frozen) != runtime['source_sha256']:
        raise ValueError('Source changed during account pool launch')
    env['CODE_ROOT'] = env['PYTHONPATH'] = str(frozen)
    state = dict(schema='robodojo.account_pool.v1', state='running',
                 config=str(pool.config_path), resume_identity=runtime,
                 planned_episodes=len(jobs), jobs=[])
    def save():
        write_json(directory/'pool_run.json', state)
        if collection and state['jobs']:
            current_job = state['jobs'][-1]
            current_case = next(c for c in cases if c['case_id'] == current_job['case_id'])
            save_sample_summary(env, current_case, current_job)
        if env.get('ROLLOUT_WORKER_STATUS_PATH'):
            job = state['jobs'][-1] if state['jobs'] else {}
            write_json(env['ROLLOUT_WORKER_STATUS_PATH'], dict(worker=env.get('ROLLOUT_WORKER_ID'),
                state=state['state'], case_id=job.get('case_id'), repeat_id=job.get('repeat_id'),
                job_index=job.get('job_index'), waiting=state.get('waiting'),
                pool_run=str(directory/'pool_run.json'), updated_at=time.time()))
    def waiting(value):
        state.update(state='waiting_for_accounts', waiting=value)
        if state['jobs']:
            state['jobs'][-1]['state'] = 'waiting_for_accounts'
        save()
    save()
    old_handlers = {}
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Account pool stopped by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    any_timeout = False
    any_infrastructure_failure = False
    try:
        selected_jobs = job_source() if job_source else enumerate(jobs)
        for job_index, (repeat_id, case) in selected_jobs:
            if cancelled():
                raise WorkerRetired('Worker draining; return pending sampling item')
            job = dict(case_id=case['case_id'], repeat_id=repeat_id,
                       job_index=job_index, state='pending', attempts=[])
            state['jobs'].append(job)
            if env['ROLLOUT_RUN_MODE'] == 'resume':
                if collection:
                    previous = collection_completed(replica_root, runtime, case['case_id'], repeat_id)
                    prior = previous['archive'] if previous else None
                    any_timeout |= bool(previous and previous.get('failure_kind') == 'wall_time_limit')
                    any_infrastructure_failure |= bool(previous and previous.get('failure_kind') == 'model_recovery_exhausted')
                else:
                    prior = completed_archive(Path(env['ROLLOUT_SHARED_ROOT']), env['ROLLOUT_EXPERIMENT_ID'],
                        case, env['ROLLOUT_REPLICA_ID'], case_identity(panel, case), runtime)
                if prior:
                    job.update(state='skipped_completed', archive=str(prior))
                    emit('rollout_skip', case_id=case['case_id'], repeat_id=repeat_id,
                         job_index=job_index, archive=str(prior))
                    save()
                    continue
            connection_failures = 0
            retry_wait = False
            while True:
                if retry_wait:
                    connection_recovery.wait_before_retry(cancelled)
                    retry_wait = False
                if cancelled():
                    raise WorkerRetired('Worker draining; return pending sampling item')
                label = f'{case["case_id"]} repeat={repeat_id} job_index={job_index}'
                with pool.acquire(label, on_wait=waiting) as account:
                    if cancelled():
                        raise WorkerRetired('Worker draining; return pending sampling item')
                    numbers = [int(p.name[8:]) for p in replica_root.glob('attempt_*') if p.name[8:].isdigit()]
                    attempt = str(allocate_attempt() if allocate_attempt else max(numbers, default=-1) + 1)
                    metadata = {k: account[k] for k in ('auth_profile', 'account_fingerprint')}
                    if env.get('ROLLOUT_WORKER_ID'):
                        metadata['worker_id'] = env['ROLLOUT_WORKER_ID']
                    child_env = dict(env, ROLLOUT_COUNT='1', ROLLOUT_CASE_ID=case['case_id'],
                        ROLLOUT_CONNECTION_EPISODE_RETRY=str(connection_failures),
                        ROLLOUT_MODEL_EPISODE_RETRY=str(int(model_retry_used(
                            replica_root, runtime, case['case_id'], repeat_id))),
                        ROLLOUT_CASE_IDS='', ROLLOUT_ATTEMPT=attempt, ROLLOUT_ALLOW_SEQUENTIAL_BATCH='1',
                        ROLLOUT_SKIP_COMPLETED_CASES='0', ROLLOUT_AUTH_PROFILE=account['auth_profile'],
                        ROLLOUT_CODEX_HOME_DIR=str(account_home(account['auth_profile'], env['ROLLOUT_AUTH_ROOT'])),
                        ROLLOUT_ACCOUNT_LOCK_FD=str(account['fds'][0]),
                        ROLLOUT_ACCOUNT_FINGERPRINT=account['account_fingerprint'])
                    if collection:
                        from .evaluation import reserve_sample_attempt
                        # Batch identity is indexed in the frozen full task panel,
                        # not in a subset or the cross-task dispatch queue.
                        from .collection import sampling_job_index
                        sample_attempt = reserve_sample_attempt(env['ROLLOUT_SHARED_ROOT'],
                            env['ROLLOUT_EXPERIMENT_ID'], case, env['ROLLOUT_REPLICA_ID'], repeat_id)
                        child_env.update(ROLLOUT_COLLECTION_REPEAT_ID=str(repeat_id),
                                         ROLLOUT_SAMPLE_ATTEMPT=str(sample_attempt),
                                         ROLLOUT_COLLECTION_JOB_INDEX=str(sampling_job_index(panel, case['case_id'], repeat_id)))
                    record = dict(**metadata, attempt=attempt, state='running', started_at=time.time(),
                                  connection_episode_retry=connection_failures)
                    record['batch_attempt'] = attempt
                    if collection:
                        record['sample_attempt'] = sample_attempt
                    job['attempts'].append(record)
                    job['state'] = state['state'] = 'running'
                    state.pop('waiting', None)
                    save()
                    code = runner(child_env, account['fds'])
                    batch = load_json(replica_root/f'attempt_{attempt}'/'batch.json')
                    rows = batch.get('episodes', [])
                    if len(rows) != 1 or rows[0].get('case_id') != case['case_id']:
                        raise RuntimeError('Missing or mismatched episode receipt; no automatic retry')
                    row = rows[0]
                    if collection and row.get('repeat_id') != repeat_id:
                        raise RuntimeError('Collection repeat receipt mismatch')
                    record.update(archive=row['archive'], exit_code=code, finished_at=time.time(),
                                  state=row.get('state', 'failed'), failure_kind=row.get('failure_kind'))
                    if not clean_result(batch, row):
                        raise RuntimeError('Local/remote cleanup not confirmed; no account switch allowed')
                    reason = row.get('failure_kind')
                    if code != 0 and reason in (None, 'model_recovery_exhausted'):
                        # Old frozen batch.py cannot classify a newly understood
                        # infrastructure error. Annotate this dispatcher receipt only;
                        # never mutate the frozen source or prior batch artifacts.
                        recovered_kind = archived_failure_kind(row)
                        if recovered_kind in connection_recovery.RETRYABLE_EPISODE_FAILURES:
                            reason = record['failure_kind'] = recovered_kind
                            record['failure_kind_source'] = 'dispatcher_controller_failure'
                            emit('episode_failure_classified',case_id=case['case_id'],
                                 failure_kind=reason,source='controller/failure.json')
                    if code != 0 and reason in ('quota_interrupted', 'auth_interrupted'):
                        record['state'] = reason
                        pool.block(account['account_fingerprint'], reason, account['credential_revision'])
                        emit('account_pool_switch', case_id=case['case_id'], **metadata,
                             reason=reason, interrupted_attempt=attempt)
                        save()
                        continue
                    if code == 0 and row.get('state') == 'completed' and reason == 'teacher_abandon':
                        from .skill_auto_infer.abandon import acknowledged
                        if not (runtime.get('teacher_abandon') and row.get('artifact_status') == 'verified'
                                and acknowledged(row.get('result') or {}, row.get('evaluation_outcome') or {})):
                            raise RuntimeError('Unverified abandonment receipt')
                        job['state'] = record['state'] = 'teacher_abandoned'
                        if recovery is not None:
                            recovery.completed(worker_index)
                    elif code == 0 and row.get('state') == 'completed':
                        job['state'] = record['state'] = 'completed'
                        if recovery is not None:
                            recovery.completed(worker_index)
                    elif code != 0 and reason in connection_recovery.RETRYABLE_EPISODE_FAILURES and recovery is not None:
                        connection_failures += 1
                        health = recovery.failed(worker_index)
                        transport_name = ('simulator' if reason == 'simulator_connection_failed' else
                                          'codex_compaction' if reason == 'codex_compaction_failed' else 'codex')
                        failure_event = ('codex_compaction_failure' if reason == 'codex_compaction_failed'
                                         else f'{transport_name}_connection_failure')
                        emit(failure_event, failure_kind=reason, worker=env.get('ROLLOUT_WORKER_ID'),
                             case_id=case['case_id'], attempt=attempt, **health)
                        if connection_failures <= connection_recovery.EPISODE_RETRIES and not cancelled():
                            job['state'] = state['state'] = 'waiting_connection_retry'
                            emit(f'{transport_name}_episode_retry', case_id=case['case_id'], repeat_id=repeat_id,
                                 interrupted_attempt=attempt, retry=connection_failures,
                                 delay_seconds=connection_recovery.RETRY_DELAY_SECONDS, reset_from_start=True)
                            save()
                            retry_wait = True
                            continue
                        job['state'] = 'pending_recollection'
                        exhausted_reason = ('codex_compaction_retry_exhausted' if reason == 'codex_compaction_failed'
                                            else f'{transport_name}_connection_retry_exhausted')
                        job['pending_reason'] = health['pause_reason'] or exhausted_reason
                        recovery.defer(job_index)
                        any_infrastructure_failure = True
                    elif reason == 'wall_time_limit' and (collection or row.get('artifact_status') == 'verified'):
                        job['state'] = record['state'] = 'wall_time_limit'
                        any_timeout = True
                    elif code != 0 and reason == 'model_recovery_exhausted':
                        if child_env['ROLLOUT_MODEL_EPISODE_RETRY'] == '0':
                            emit('model_episode_retry', case_id=case['case_id'], repeat_id=repeat_id,
                                 interrupted_attempt=attempt, retry=1, reset_from_start=True)
                            save()
                            continue
                        job['state'] = record['state'] = 'infrastructure_failed'
                        any_infrastructure_failure = True
                    else:
                        raise RuntimeError(f'Episode failed (exit {code}); not an account failure, queue stopped')
                    save()
                    emit('collection_job_done' if collection else 'account_pool_job_done',
                         job_index=job_index, planned_episodes=len(jobs), repeat_id=repeat_id,
                         case_id=case['case_id'], state=job['state'], archive=row['archive'])
                    break
            if recovery is not None and recovery.stop.is_set():
                break
        state['state'] = ('completed_with_infrastructure_failures' if any_infrastructure_failure
                          else 'completed_with_timeouts' if any_timeout else 'completed')
        if recovery is not None:
            state['connection_recovery'] = recovery.snapshot()
            if recovery.stop.is_set():
                state['state'] = 'dispatch_paused_connection_errors'
        save()
        emit('account_pool_done', state=state['state'], summary=str(directory/'pool_run.json'))
        if recovery is not None and recovery.stop.is_set() and not env.get('ROLLOUT_WORKER_ID'):
            return 1
        return 124 if any_timeout else 0
    except BaseException as error:
        if isinstance(error, WorkerRetired) and recovery is not None and state['jobs']:
            current_job = state['jobs'][-1]
            attempts = current_job['attempts']
            if attempts and attempts[-1].get('failure_kind') in connection_recovery.RETRYABLE_EPISODE_FAILURES:
                current_job.update(state='pending_recollection', pending_reason='dispatch_stopped_before_retry')
                recovery.defer(current_job['job_index'])
            state['connection_recovery'] = recovery.snapshot()
        if isinstance(error, WorkerRetired) and env.get('ROLLOUT_WORKER_ID'):
            state.update(state='retired', error_type=type(error).__name__)
            save()
            if raise_on_retire:
                raise
            return 124 if any_timeout else 0
        state.update(state='stopped' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error_type=type(error).__name__)
        save()
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

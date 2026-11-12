"""Resizable Linux workers; one owner of campaign/resources and job allocation.

Worker death never requeues an ambiguous episode in the same run. A confirmed
clean simulator disconnect permits one new episode attempt in account_pool.
Resume uses verified archived receipts, not in-memory queue state. No policy changes.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import uuid

from .account_pool import AccountPool, WorkerRetired, emit, run_pool, load_json
from .io import write_json
from .ssh_connection import simulator_user

CONFIG_POLL_SECONDS = 5
SUPERVISOR_POLL_SECONDS = 1


def slots(env):
    if env.get('ROLLOUT_WORKER_SLOTS_JSON'):
        return explicit_slots(env)
    if env.get('ROLLOUT_STRICT_EVAL') == '1' and not env.get('ROLLOUT_WORKER_CONTROL_FILE'):
        count = int(env.get('ROLLOUT_PARALLEL_WORKERS', '1'))
        if not 1 <= count <= 8:
            raise ValueError('Eval supports 1..8 workers')
        if count > 1:
            if not env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
                raise ValueError('Parallel eval requires an account pool')
            # Reuse collection's eight physical slots, without enabling any
            # collection policy, step budget or dynamic control file.
            configured = slots(dict(env, ROLLOUT_WORKER_CONTROL_FILE='eval-slot-layout-only'))[:count]
            for slot in configured:
                slot.pop('ROLLOUT_WORKER_CONTROL_FILE')
            return configured
    if env.get('ROLLOUT_WORKER_CONTROL_FILE'):
        if not env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
            raise ValueError('Dynamic collection requires an account pool')
        result = []
        for i in range(8):
            # 0..3: one simulator/GPU. 4..7: second simulator on those GPUs.
            host_index, gpu, lane = i % 2, (i % 4)//2, i//4
            e = dict(env, ROLLOUT_WORKER_ID=f'worker_{i}', SIM_GPU=str(gpu),
                POLICY_PORT=str(int(env['POLICY_PORT'])+i),
                SIM_PORT=str(int(env['SIM_PORT'])+i),
                ROLLOUT_SIM_REMOTE_PORT=str(int(env['ROLLOUT_SIM_REMOTE_PORT'])+gpu+2*lane))
            if host_index:
                e.update(ROLLOUT_SIM_HOST=env.get('ROLLOUT_SECOND_SIM_HOST', env['ROLLOUT_SIM_HOST']),
                         ROLLOUT_SIM_SSH_PORT=env.get('ROLLOUT_SECOND_SIM_SSH_PORT', '5727'))
            result.append(e)
        ports = [int(e[k]) for e in result for k in ('POLICY_PORT','SIM_PORT')]
        remote = [(e['ROLLOUT_SIM_HOST'],e['ROLLOUT_SIM_SSH_PORT'],e['ROLLOUT_SIM_REMOTE_PORT']) for e in result]
        if (len(set(ports)) != 16 or any(not 1024 <= p <= 65535 for p in ports)
                or len(set(remote)) != 8
                or any(not 1024 <= int(e['ROLLOUT_SIM_REMOTE_PORT']) <= 65535 for e in result)):
            raise ValueError('Dynamic slots require distinct valid ports and two different hosts')
        return result
    count = int(env.get('ROLLOUT_PARALLEL_WORKERS', '1'))
    if count == 1:
        return [dict(env)]
    if count != 2:
        raise ValueError('Only 1 or 2 workers supported in this version')
    if not env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
        raise ValueError('Parallel collection requires an account pool')
    second = dict(env, ROLLOUT_SIM_SSH_PORT=env.get('ROLLOUT_SECOND_SIM_SSH_PORT', '5727'),
        ROLLOUT_SIM_HOST=env.get('ROLLOUT_SECOND_SIM_HOST', env['ROLLOUT_SIM_HOST']),
        SIM_GPU=env.get('ROLLOUT_SECOND_SIM_GPU', '0'),
        POLICY_GPU=env.get('ROLLOUT_SECOND_POLICY_GPU', env.get('POLICY_GPU', '0')),
        POLICY_PORT=env.get('ROLLOUT_SECOND_POLICY_PORT', '18831'),
        SIM_PORT=env.get('ROLLOUT_SECOND_SIM_PORT', '19114'),
        ROLLOUT_SIM_REMOTE_PORT=env.get('ROLLOUT_SECOND_REMOTE_PORT', '19113'))
    result = [dict(env), second]
    ports = [int(e[k]) for e in result for k in ('POLICY_PORT', 'SIM_PORT')]
    if len(set(ports)) != 4 or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError('Four distinct A100 policy/tunnel ports required')
    resources = [(e['ROLLOUT_SIM_HOST'], e['ROLLOUT_SIM_SSH_PORT'], e['SIM_GPU']) for e in result]
    if resources[0] == resources[1]:
        raise ValueError('Workers must use distinct remote GPU slots')
    remote_ports = [(e['ROLLOUT_SIM_HOST'],e['ROLLOUT_SIM_SSH_PORT'],e['ROLLOUT_SIM_REMOTE_PORT']) for e in result]
    if remote_ports[0] == remote_ports[1]:
        raise ValueError('Workers on the same host need distinct remote ports')
    for i, e in enumerate(result):
        e['ROLLOUT_WORKER_ID'] = f'worker_{i}'
    return result


def explicit_slots(env):
    """User-defined physical slots; validate addresses, never infer GPU capacity."""
    values = json.loads(env['ROLLOUT_WORKER_SLOTS_JSON'])
    if not isinstance(values, list) or not values:
        raise ValueError('worker_slots must be a nonempty list')
    required = {'ROLLOUT_SIM_HOST', 'ROLLOUT_SIM_SSH_PORT', 'SIM_GPU', 'POLICY_GPU',
                'POLICY_PORT', 'SIM_PORT', 'ROLLOUT_SIM_REMOTE_PORT'}
    optional = {'ROLLOUT_SIM_SSH_USER', 'ROLLOUT_SIM_SSH_KEY', 'ROLLOUT_SIM_ROBODOJO_SOURCE',
                'ROLLOUT_SIM_PYTHON', 'ROLLOUT_SIM_ENV_SCRIPT'}
    result = []
    for i, row in enumerate(values):
        if not isinstance(row, dict) or not required <= row.keys() or row.keys() - required - optional:
            raise ValueError('worker_slots: missing or unsupported fields')
        if any(not isinstance(v, (str, int)) or isinstance(v, bool) or not str(v).strip() for v in row.values()):
            raise ValueError('worker_slots: expected nonempty strings or integers')
        e = dict(env, **{k: str(v) for k, v in row.items()}, ROLLOUT_WORKER_ID=f'worker_{i}')
        simulator_user(e)
        for key in ('POLICY_PORT', 'SIM_PORT', 'ROLLOUT_SIM_REMOTE_PORT', 'ROLLOUT_SIM_SSH_PORT'):
            if not 1 <= int(e[key]) <= 65535:
                raise ValueError('worker_slots: invalid port')
        result.append(e)
    local = [e[k] for e in result for k in ('POLICY_PORT', 'SIM_PORT')]
    remote = [(e['ROLLOUT_SIM_HOST'], e['ROLLOUT_SIM_SSH_PORT'], e['ROLLOUT_SIM_REMOTE_PORT']) for e in result]
    if len(set(local)) != len(local) or len(set(remote)) != len(remote):
        raise ValueError('worker_slots: duplicate policy/tunnel/remote ports')
    count = int(env.get('ROLLOUT_PARALLEL_WORKERS', '1'))
    if not 1 <= count <= len(result):
        raise ValueError('Requested more workers than configured slots')
    if (len(result) > 1 or count > 1) and not env.get('ROLLOUT_ACCOUNT_POOL_FILE'):
        raise ValueError('Parallel execution requires an account pool')
    # Collection retains all slots for live resize; strict eval uses the requested count.
    return result if env.get('ROLLOUT_WORKER_CONTROL_FILE') else result[:count]


class Allocation:
    def __init__(self, ctx, jobs, first_attempt):
        self.jobs = jobs
        self.cursor = ctx.Value('i', 0)
        self.attempt = ctx.Value('q', first_attempt)
        self.stop = ctx.Event()
        self.states = ctx.Array('i', len(jobs))  # 0 pending, 1 claimed, 2 acknowledged by worker
        from .connection_recovery import ConnectionRecovery
        self.connection_recovery = ConnectionRecovery(ctx, self.stop, len(jobs))

    def source(self, retire=None):
        retire = retire or self.stop
        while not self.stop.is_set():
            with self.states.get_lock():
                index = next((i for i, state in enumerate(self.states) if state == 0), None)
                if index is None or self.stop.is_set() or retire.is_set():
                    return
                self.states[index] = 1
            acknowledged = False
            try:
                yield index, self.jobs[index]
                acknowledged = True
            finally:
                with self.states.get_lock():
                    self.states[index] = 2 if acknowledged else 0

    def pending(self):
        with self.states.get_lock():
            return any(s == 0 for s in self.states)

    def complete(self):
        with self.states.get_lock():
            return all(s == 2 for s in self.states)

    def allocate_attempt(self):
        with self.attempt.get_lock():
            number = self.attempt.value
            self.attempt.value += 1
            return number


SLOT_KEYS = ('ROLLOUT_WORKER_ID', 'ROLLOUT_SIM_HOST', 'ROLLOUT_SIM_SSH_PORT',
             'SIM_GPU', 'POLICY_GPU', 'POLICY_PORT', 'SIM_PORT', 'ROLLOUT_SIM_REMOTE_PORT',
             'ROLLOUT_SIM_SSH_USER', 'ROLLOUT_SIM_SSH_KEY', 'ROLLOUT_SIM_ROBODOJO_SOURCE', 'ROLLOUT_SIM_PYTHON',
             'ROLLOUT_SIM_ENV_SCRIPT',
             'ROLLOUT_RESOURCE_LOCK_FDS', 'ROLLOUT_WORKER_STATUS_PATH', 'CODE_ROOT', 'PYTHONPATH')


def suite_jobs(campaigns):
    """Round-robin tasks, with no task-completion barrier."""
    return [(task, (None if env.get('ROLLOUT_STRICT_EVAL') == '1' else 1, cases[index]))
            for index in range(max(len(cases) for _, cases in campaigns))
            for task, (env, cases) in enumerate(campaigns) if index < len(cases)]


def run_suite_jobs(env, campaigns, source, allocation, pool):
    code = 0
    for index, (task, job) in source:
        task_env, cases = campaigns[task]
        child = dict(task_env, **{k:env[k] for k in SLOT_KEYS if k in env})
        child['ROLLOUT_SUITE_JOB_INDEX'] = str(index)
        emit('collection_suite_dispatch', worker=env['ROLLOUT_WORKER_ID'],
             task=child['ROBODOJO_TASK'], experiment=child['ROLLOUT_EXPERIMENT_ID'],
             checkpoint=child.get('CHECKPOINT'), prompt_profile=child.get('ROLLOUT_PROMPT_PROFILE'),
             case_id=job[1]['case_id'], job_index=index)
        result = run_pool(child, cases, pool=pool,
                          job_source=lambda: iter([(index, job)]),
                          allocate_attempt=allocation.allocate_attempt, raise_on_retire=True)
        if result not in (0, 124):
            return result
        code = max(code, result)
    return code


def worker(env, cases, allocation, directory, retire=None, campaigns=None):
    name = env['ROLLOUT_WORKER_ID']
    env = dict(env, ROLLOUT_WORKER_STATUS_PATH=str(directory/f'{name}.live.json'))
    log = os.open(directory/f'{name}.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(log, 1); os.dup2(log, 2); os.close(log)
    code = 1
    retire = retire or allocation.stop
    source = allocation.source(retire)
    try:
        pool = AccountPool(env)
        pool.connection_recovery = allocation.connection_recovery
        pool.cancelled = lambda: allocation.stop.is_set() or retire.is_set()
        if campaigns is None:
            code = run_pool(env, cases, pool=pool, job_source=lambda: source,
                            allocate_attempt=allocation.allocate_attempt)
        else:
            code = run_suite_jobs(env, campaigns, source, allocation, pool)
    except WorkerRetired:
        if allocation.connection_recovery.snapshot()['pause_reason']:
            code = 0
            emit('worker_retired', worker=name, reason='connection_circuit_breaker')
        elif retire.is_set() and not allocation.stop.is_set():
            code = 0
            emit('worker_retired', worker=name, reason='scale_down_before_next_episode')
        else:
            allocation.stop.set()
    except BaseException as error:
        allocation.stop.set()
        emit('parallel_worker_error', worker=name, error_type=type(error).__name__, error=str(error))
    finally:
        source.close()  # Return a claimed-but-not-started job on retirement.
        write_json(directory/f'{name}.json', dict(worker=name, exit_code=code, finished_at=time.time()))
    raise SystemExit(code)


def read_desired(path):
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError('Worker control must be a JSON object')
    desired = value.get('desired_workers')
    if type(desired) is not int or desired < 0:
        raise ValueError('desired_workers must be a nonnegative integer (0 pauses dispatch)')
    return desired


def dispatch_state(stopped, desired, active):
    if stopped:
        return 'draining_after_error' if active else 'failed'
    return ('draining' if active else 'paused') if desired == 0 else 'running'


def run_parallel(environments, cases, *, campaigns=None, output=None):
    from .evaluation import read_panel
    from .collection import SCHEMA
    from .deployment import snapshot
    env = environments[0]
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'])
    from .regression_eval import SCHEMA as EVAL_SCHEMA, runtime_settings
    collection = panel.get('schema') == SCHEMA
    if not collection:
        if panel.get('schema') != EVAL_SCHEMA:
            raise ValueError('Parallel mode supports collection or regression eval manifests only')
        for worker_env in environments:
            runtime_settings(worker_env, panel)
    ctx = mp.get_context('fork')
    base = Path(env['ROLLOUT_SHARED_ROOT'])/'results'/env['ROLLOUT_EXPERIMENT_ID']
    directory = (Path(output) if output else base/'_parallel')/f'run_{uuid.uuid4().hex}'
    directory.mkdir(parents=True)
    identity = json.loads(env['ROLLOUT_RESUME_IDENTITY'])
    frozen = directory/'source_snapshot'
    if snapshot(env['CODE_ROOT'], frozen) != identity['source_sha256']:
        raise ValueError('Source changed while freezing parallel campaign')
    root = base/'_replicas'/f'replica_{env["ROLLOUT_REPLICA_ID"]}'
    numbers = [int(p.name[8:]) for p in root.glob('attempt_*') if p.name[8:].isdigit()]
    jobs = ([(r, c) for r in range(1, panel['collection']['repeats']+1) for c in cases]
            if collection else [(None, c) for c in cases])
    if campaigns is not None:
        jobs = suite_jobs(campaigns)
        numbers = []
        for campaign, _ in campaigns:
            if json.loads(campaign['ROLLOUT_RESUME_IDENTITY'])['source_sha256'] != identity['source_sha256']:
                raise ValueError('Suite campaigns must share one source snapshot')
            root = Path(campaign['ROLLOUT_SHARED_ROOT'])/'results'/campaign['ROLLOUT_EXPERIMENT_ID']
            root = root/'_replicas'/f'replica_{campaign["ROLLOUT_REPLICA_ID"]}'
            numbers.extend(int(p.name[8:]) for p in root.glob('attempt_*') if p.name[8:].isdigit())
    allocation = Allocation(ctx, jobs, max(numbers, default=-1)+1)
    children = []
    active = {}
    generations = [0]*len(environments)
    control = env.get('ROLLOUT_WORKER_CONTROL_FILE')
    desired = read_desired(control) if control else len(environments)
    if desired > len(environments):
        raise ValueError('Requested more workers than configured slots')
    status = dict(state='running', planned_episodes=len(jobs), resume_identity=identity,
                  workers=[], started_at=time.time(), control_file=control,
                  slots=[{k:e.get(k) for k in ('ROLLOUT_WORKER_ID','ROLLOUT_SIM_HOST',
                      'ROLLOUT_SIM_SSH_PORT','ROLLOUT_SIM_SSH_USER','SIM_GPU','POLICY_GPU','POLICY_PORT','SIM_PORT',
                      'ROLLOUT_SIM_REMOTE_PORT')} for e in environments])
    if campaigns is not None:
        status['campaigns'] = [dict(task=e['ROBODOJO_TASK'], experiment=e['ROLLOUT_EXPERIMENT_ID'],
                                   resume_identity=json.loads(e['ROLLOUT_RESUME_IDENTITY'])) for e,_ in campaigns]
        status['queue'] = [dict(job_index=i, task=campaigns[t][0]['ROBODOJO_TASK'],
                               experiment=campaigns[t][0]['ROLLOUT_EXPERIMENT_ID'],
                               case_id=job[1]['case_id']) for i,(t,job) in enumerate(jobs)]
    def save():
        if status['state'] in ('running', 'draining', 'paused', 'draining_after_error'):
            status['state'] = dispatch_state(allocation.stop.is_set(), desired, bool(active))
        status['dispatch_stopped'] = allocation.stop.is_set()
        status['connection_recovery'] = allocation.connection_recovery.snapshot()
        status['desired_workers'] = desired
        status['job_states'] = list(allocation.states)
        status['workers'] = [dict(worker=f'worker_{i}', generation=g, pid=p.pid, exit_code=p.exitcode,
                                 draining=retire.is_set(), log=str(d/f'worker_{i}.log'),
                                 progress=load_json(d/f'worker_{i}.live.json'))
                             for i,g,p,retire,d in children]
        write_json(directory/'status.json', status)
        if campaigns is not None and output is not None:
            write_json(Path(output).parent/'status.json', dict(state=status['state'],
                planned_episodes=len(jobs), dispatch_stopped=status['dispatch_stopped'],
                queue_root=str(output), queue_status=str(directory/'status.json'),
                desired_workers=desired, active_workers=len(active),
                connection_recovery=status['connection_recovery'],
                failed_workers=[w['worker'] for w in status['workers']
                                if w['exit_code'] not in (None, 0, 124)]))
    def stop(signum, frame):
        raise KeyboardInterrupt()
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        save()
        emit('parallel_started', directory=str(directory), workers=status['workers'])
        last = checked = 0
        while True:
            if any(p.exitcode not in (None, 0, 124) for _,_,p,_,_ in children):
                allocation.stop.set()  # Other running episode may finish; no new work.
            if allocation.stop.is_set() and not status.get('dispatch_stopped'):
                save()
                emit('parallel_dispatch_stopped', state=status['state'],
                     reason=(allocation.connection_recovery.snapshot()['pause_reason'] or
                             'Worker error; active episodes may finish, no new jobs will start'),
                     status_file=str(directory/'status.json'))
            for i, (p, retire) in list(active.items()):
                if not p.is_alive():
                    p.join()
                    del active[i]
            if allocation.stop.is_set() and not active:
                break
            if allocation.complete() and not active:
                break
            if control and time.monotonic()-checked >= CONFIG_POLL_SECONDS:
                try:
                    new = read_desired(control)
                    if new > len(environments):
                        raise ValueError('Requested more workers than configured slots')
                    if new != desired:
                        emit('parallel_resize', previous=desired, desired_workers=new,
                             note='Scale-down drains current episode; never interrupts it')
                        desired = new
                    status.pop('control_error', None)
                except (ValueError, OSError, TypeError) as error:
                    status['control_error'] = str(error)
                    emit('parallel_config_invalid', error=str(error), retained_workers=desired)
                checked = time.monotonic()
                save()
            for i, (_, retire) in active.items():
                if i >= desired:
                    retire.set()
            if not allocation.stop.is_set() and allocation.pending():
                for i in range(desired):
                    if i in active:
                        continue
                    generations[i] += 1
                    d = directory/f'worker_{i}_generation_{generations[i]}'
                    d.mkdir()
                    e = dict(environments[i], CODE_ROOT=str(frozen), PYTHONPATH=str(frozen))
                    retire = ctx.Event()
                    p = ctx.Process(target=worker, args=(e, cases, allocation, d, retire, campaigns))
                    p.start()
                    active[i] = p, retire
                    children.append((i,generations[i],p,retire,d))
                    emit('parallel_worker_started', worker=i, generation=generations[i], directory=str(d))
            if time.monotonic()-last >= 60:
                save()
                emit('parallel_heartbeat', workers=status['workers'], dispatch_stopped=allocation.stop.is_set())
                last = time.monotonic()
            time.sleep(SUPERVISOR_POLL_SECONDS)
        codes = [p.exitcode for _,_,p,_,_ in children]
        code = 1 if allocation.stop.is_set() or any(c not in (0,124) for c in codes) else 124 if 124 in codes else 0
        status['state'] = ('failed' if code == 1 else 'completed_with_timeouts' if code else 'completed')
        health = allocation.connection_recovery.snapshot()
        if health['pause_reason']:
            status['state'] = 'dispatch_paused_connection_errors'
        elif health['pending_recollection_job_indices'] and code != 1:
            status['state'] = 'completed_with_pending_recollection'
        return code
    except BaseException:
        status['state'] = 'interrupted'
        allocation.stop.set()
        for _,_,p,_,_ in children:
            if p.is_alive(): p.terminate()
        raise
    finally:
        # Do not release resource locks while a worker is performing cleanup.
        for _,_,p,_,_ in children:
            while p.is_alive():
                p.join(timeout=5)
                emit('parallel_cleanup_wait', worker_pid=p.pid)
            p.join()
        save()
        for sig, handler in old.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Atomically set dynamic collection concurrency; no model calls')
    parser.add_argument('--config', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set-workers', type=int)
    group.add_argument('--init', action='store_true')
    args = parser.parse_args()
    path = Path(args.config)
    if args.init and path.exists():
        print(f'Existing control preserved: {path}; desired_workers={read_desired(path)}')
    else:
        if args.set_workers is not None and args.set_workers < 0:
            parser.error('--set-workers must be nonnegative')
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, dict(desired_workers=2 if args.init else args.set_workers))
        print(f'{path}: desired_workers={read_desired(path)}')

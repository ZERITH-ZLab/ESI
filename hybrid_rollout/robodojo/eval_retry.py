"""Bounded recovery for a frozen queue or pilot + supplement, including live attachment."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .account_pool import clean_result, archived_failure_kind, load_json
from .connection_recovery import RETRYABLE_EPISODE_FAILURES
from .eval_followup import aggregate, summaries, validate_partition
from .eval_sequence import read_config
from .io import require, write_json

RETRYABLE = RETRYABLE_EPISODE_FAILURES | {'wall_time_limit'}
DELAYS = (60, 300, 900)


def busy(path):
    if not path.exists():
        return False
    # NFS implements flock with a write lock; LOCK_EX needs a writable fd.
    # r+ preserves the existing lock file without truncation or creation.
    with path.open('r+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def receipt_reason(batch, row):
    if not clean_result(batch, row):
        return None
    reason = archived_failure_kind(row)
    return reason if reason in RETRYABLE else None


def assess_queue(shared, cfg, reports):
    """Fail closed on user interruption, unknown crashes, or unconfirmed cleanup."""
    reasons, blocked = Counter(), []
    directory = shared/'evaluation'/cfg['name']
    status = load_json(directory/'status.json')
    detail = load_json(status['queue_status']) if status.get('queue_status') else status
    if detail.get('state') in ('interrupted', 'running', 'paused', 'draining', 'draining_after_error'):
        blocked.append('queue_not_terminal_or_operator_interrupted')
    by_experiment = {r['experiment']: r for r in reports}
    for model in cfg['models']:
        report = by_experiment[model['experiment']]
        latest = {}
        paths = (shared/'results'/model['experiment']).glob('_replicas/replica_*/attempt_*/batch.json')
        for path in sorted(paths, key=lambda p: int(p.parent.name.removeprefix('attempt_'))):
            batch = load_json(path)
            for row in batch.get('episodes', []):
                if row.get('state') != 'skipped_completed':
                    latest[row['case_id']] = (batch, row)
        for episode in report['episodes']:
            if episode['state'] in ('completed','excluded'):
                continue  # Native success AND native failure are final evaluations.
            if episode['state'] == 'not_started':
                continue
            pair = latest.get(episode['case_id'])
            if (pair and pair[1].get('state') == 'completed'
                    and (pair[1].get('evaluation_outcome') or {}).get('status') == 'invalid_native_layout'):
                blocked.append(episode['case_id'] + ': invalid_native_layout; decide invalid-case handling, not a transport retry')
                continue
            reason = receipt_reason(*pair) if pair and episode['state']=='incomplete' else None
            if reason:
                reasons[reason] += 1
            else:
                blocked.append(episode['case_id'] + ': unknown error, duplicate, or cleanup unconfirmed')
    # A worker can crash before producing any episode receipt. Do not hide it
    # just because other cases happen to contain recoverable failures.
    for worker in detail.get('workers', []):
        if worker.get('exit_code') in (0, 124):
            continue
        events = []
        for line in Path(worker['log']).read_text(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('event') == 'parallel_worker_error':
                events.append(event)
        expected = 'Episode failed (exit 1); not an account failure, queue stopped'
        progress = worker.get('progress') or {}
        pool = load_json(progress['pool_run']) if progress.get('pool_run') else {}
        jobs = pool.get('jobs', [])
        attempts = jobs[-1].get('attempts', []) if jobs else []
        record = attempts[-1] if attempts else {}
        reason = archived_failure_kind(record)
        if not events or events[-1].get('error') != expected or reason not in RETRYABLE:
            blocked.append(worker['worker'] + ': unclassified worker exit')
    if status.get('state') == 'failed' and not reasons and not blocked:
        blocked.append('queue failed without a recoverable episode receipt')
    return dict(retryable=not blocked, reasons=dict(reasons), blocked=blocked)


def run_command(argv, env, log):
    with log.open('a') as stream:
        child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            return child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                child.wait()  # Owned workers must complete their cleanup.
            raise


def run(path, env, *, max_restarts=3):
    require(type(max_restarts) is int and 0 <= max_restarts <= 10, 'max_restarts must be 0..10')
    path = Path(path).resolve()
    cfg = load_json(path)
    single = cfg.get('schema') == 'robodojo.eval_sequence.v1'
    if single:
        from .eval_queue import validate_queue
        first_path = second_path = path
        first = second = read_config(path)
        validate_queue(first)
        # The queue freezes current release code on first launch and verifies its
        # experiment snapshot on resume. A historical external pin is optional.
    else:
        require(cfg.get('schema') == 'robodojo.eval_followup.v1', 'Expected queue or followup config')
        first_path, second_path = [path.parent/cfg[k] for k in ('predecessor_config','successor_config')]
        first, second = read_config(first_path), read_config(second_path)
        validate_partition(first, second)
    shared = Path(env['ROLLOUT_SHARED_ROOT'])
    output = shared/'evaluation'/cfg['name']
    output.mkdir(parents=True, exist_ok=True)
    initial_pending = single and not (output/'queue_config.json').exists() and not (output/'status.json').exists()
    state = dict(schema='robodojo.eval_retry.v1', started_at=time.time(), restarts=0, initial_launches=0,
                 max_restarts=max_restarts, config=str(path))
    def save(**fields):
        state.update(fields, updated_at=time.time())
        write_json(output/'retry_status.json', state)
        print(json.dumps(dict(event='eval_retry_status', **state)), flush=True)
    locks = ([output/'queue.lock'] if single else
             [output/'followup.lock'] + [shared/'evaluation'/c['name']/'queue.lock' for c in (first,second)])
    with (output/'retry.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Eval retry supervisor already running') from None
        def interrupted(*_):
            raise KeyboardInterrupt()
        old = signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                if any(busy(p) for p in locks):
                    save(state='waiting_for_running_eval')
                    time.sleep(60)
                    continue
                a = summaries(shared, first)
                if single:
                    b = a
                    total = dict(tasks=a, **{k:sum(r[k] for r in a)
                        for k in ('planned','completed','successes','failures')})
                    total['incomplete_or_unstarted'] = total['planned']-total['completed']
                    total['original_planned'] = sum(r.get('original_planned',r['planned']) for r in a)
                    total['excluded'] = sum(r.get('excluded',0) for r in a)
                else:
                    b = summaries(shared, second)
                    b_by_task = {r['task']:r for r in b}
                    total = aggregate(a, [b_by_task[r['task']] for r in a])
                write_json(output/'summary.json', total)
                save(completed=total['completed'], planned=total['planned'])
                if total['completed'] == total['planned']:
                    save(state='completed')
                    return 0
                groups = [(first,a)] if single else [(first,a),(second,b)]
                plans = [assess_queue(shared, c, r) for c,r in groups
                         if sum(x['completed'] for x in r) < sum(x['planned'] for x in r)]
                if not all(p['retryable'] for p in plans):
                    save(state='manual_review_required', plans=plans)
                    return 1
                first_launch = initial_pending and not (output/'queue_config.json').exists() and not (output/'status.json').exists()
                if not first_launch and state['restarts'] >= max_restarts:
                    save(state='retry_budget_exhausted', plans=plans)
                    return 1
                delay = 0 if first_launch else DELAYS[min(state['restarts'],len(DELAYS)-1)]
                save(state='cooldown', delay_seconds=delay, plans=plans)
                # Respond to signals during backoff; never spin or hold GPU locks.
                deadline = time.monotonic()+delay
                while time.monotonic() < deadline:
                    time.sleep(min(30, max(0,deadline-time.monotonic())))
                if any(busy(p) for p in locks):
                    continue
                pilot_done = sum(r['completed'] for r in a) == sum(r['planned'] for r in a)
                mode, selected = (('--eval-queue',path) if single else
                    ('--eval-followup',path) if pilot_done else ('--eval-queue',first_path))
                if first_launch:
                    state['initial_launches'] += 1
                    initial_pending = False
                else:
                    state['restarts'] += 1
                launch_number = state['initial_launches'] + state['restarts']
                log = output/f'retry_{int(state["started_at"])}_{launch_number}.log'
                argv = ['bash', str(Path(env['CODE_ROOT'])/'hybrid_rollout/robodojo/launch_dual.sh'),
                        mode, '--config', str(selected)]
                queue_status = shared/'evaluation'/(second if pilot_done else first)['name']/'status.json'
                previous_run = load_json(queue_status).get('queue_status')
                save(state='resuming', log=str(log))
                code = run_command(argv, env, log)
                save(state='checking_result', exit_code=code)
                # Startup/config/identity errors must never loop on stale receipts.
                if (code not in (0,1,124) or
                        (code and load_json(queue_status).get('queue_status') == previous_run)):
                    save(state='manual_review_required', reason='launcher failed; inspect log')
                    return code or 1
        except KeyboardInterrupt:
            save(state='stopped_by_operator')
            return 130
        except Exception as error:
            save(state='manual_review_required', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            signal.signal(signal.SIGTERM, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--max-restarts',type=int,default=3)
    args = parser.parse_args()
    return run(args.config, dict(os.environ), max_restarts=args.max_restarts)


if __name__ == '__main__':
    raise SystemExit(main())

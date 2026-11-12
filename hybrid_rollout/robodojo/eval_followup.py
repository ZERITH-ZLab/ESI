"""Wait for a verified pilot, then evaluate its disjoint complement and report both."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time

from .collection_suite import task_specs
from .eval_sequence import read_config
from .eval_queue import validate_queue, execution_root, run as run_queue
from .regression_eval import report
from .io import require, write_json


def validate_partition(first, second):
    for cfg in (first, second):
        validate_queue(cfg)
    require(first['name'] != second['name'], 'Queues need distinct names')
    require(first.get('execution_source') and
            first['execution_source'] == second.get('execution_source'),
            'Both queues must pin the same episode source')
    a = {m['task']: m for m in first['models']}
    b = {m['task']: m for m in second['models']}
    require(a.keys() == b.keys(), 'Task sets differ')
    require(not ({m['experiment'] for m in a.values()} & {m['experiment'] for m in b.values()}),
            'Supplement needs separate experiment names')
    for task, model in a.items():
        other = b[task]
        require({k:v for k,v in model.items() if k not in ('cases','experiment')} ==
                {k:v for k,v in other.items() if k not in ('cases','experiment')},
                f'Policy settings differ: {task}')
        left, right = ({tuple(c) for c in m['cases']} for m in (model, other))
        expected = {(c['variant'],c['layout_id']) for c in task_specs(task)}
        require(not left & right, f'Overlapping cases: {task}')
        require(left | right == expected, f'Incomplete 20-case partition: {task}')


def summaries(shared, cfg):
    rows = []
    for model in cfg['models']:
        panel = shared/'eval_panels'/model['experiment']/'queue.json'
        if panel.exists():
            frozen = json.loads(panel.read_text())
            require(frozen['case_selection'] == model['cases'], 'Report panel changed')
            result = report(shared, model['experiment'], unified=True)
        else:
            from .eval_exclusions import unstarted_report
            result = unstarted_report(shared, model)
        rows.append(dict(result, task=model['task']))
    return rows


def aggregate(first, second):
    tasks = []
    for a, b in zip(first, second):
        require(a['task'] == b['task'], 'Report task order changed')
        row = dict(task=a['task'], experiments=[a['experiment'],b['experiment']],
                   episodes=[dict(e, source_experiment=r['experiment'])
                             for r in (a,b) for e in r['episodes']])
        for key in ('planned','completed','successes','failures'):
            row[key] = a[key]+b[key]
        row['original_planned'] = a.get('original_planned',a['planned'])+b.get('original_planned',b['planned'])
        row['excluded'] = a.get('excluded',0)+b.get('excluded',0)
        row['incomplete_or_unstarted'] = row['planned']-row['completed']
        tasks.append(row)
    result = dict(schema='robodojo.eval_combined.v1', tasks=tasks)
    for key in ('planned','completed','successes','failures','incomplete_or_unstarted'):
        result[key] = sum(t[key] for t in tasks)
    for key in ('original_planned','excluded'):
        result[key] = sum(t[key] for t in tasks)
    return result


def queue_busy(directory):
    # A terminal status alone is insufficient: wait until cleanup releases the
    # supervisor lock. Never signal or interrupt the predecessor.
    with (directory/'queue.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def predecessor_ready(rows, busy):
    return not busy and all(r['completed'] == r['planned'] for r in rows)


def run(path, env, *, dry_run=False):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    require(cfg['schema'] == 'robodojo.eval_followup.v1' and
            re.fullmatch(r'[A-Za-z0-9_-]+', cfg['name']), 'Invalid followup config')
    first = read_config(path.parent/cfg['predecessor_config'])
    second = read_config(path.parent/cfg['successor_config'])
    validate_partition(first, second)
    # Reporting uses predecessor task order regardless of successor queue order.
    by_task = {m['task']:m for m in second['models']}
    report_second = dict(second, models=[by_task[m['task']] for m in first['models']])
    execution_root(second, env)
    shared = Path(env['ROLLOUT_SHARED_ROOT'])
    prior = shared/'evaluation'/first['name']
    teacher = {k:env.get(k) for k in ('ROLLOUT_MODEL','ROLLOUT_REASONING_EFFORT',
                                     'ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT')}
    require(json.loads((prior/'queue_config.json').read_text()) == dict(config=first,teacher=teacher),
            'Predecessor frozen config/teacher differs')
    output = shared/'evaluation'/cfg['name']
    print(json.dumps(dict(event='eval_followup_plan', output=str(output),
        predecessor=first['name'], successor=second['name'], workers=second['workers'],
        pilot_cases=sum(len(m['cases']) for m in first['models']),
        additional_cases=sum(len(m['cases']) for m in second['models']))),flush=True)
    if dry_run:
        return 0
    output.mkdir(parents=True,exist_ok=True)
    with (output/'followup.lock').open('a') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Followup already running') from None
        frozen = dict(config=cfg, predecessor=first, successor=second, teacher=teacher)
        record = output/'followup_config.json'
        if record.exists():require(json.loads(record.read_text()) == frozen, 'Followup config changed')
        else:write_json(record, frozen)
        while True:
            a, b = summaries(shared,first), summaries(shared,report_second)
            write_json(output/'summary.json',aggregate(a,b))
            busy = queue_busy(prior)
            if predecessor_ready(a,busy):break
            state = dict(state='waiting_for_pilot', pilot_completed=sum(r['completed'] for r in a),
                         pilot_planned=sum(r['planned'] for r in a), predecessor_running=busy,
                         message='Waiting for all pilot cases to finish and cleanup; incomplete pilot must be resumed')
            write_json(output/'status.json',state)
            print(json.dumps(dict(event='eval_followup_waiting',**state)),flush=True)
            time.sleep(30)
        write_json(output/'status.json',dict(state='running_supplement',workers=second['workers'],
            queue_status=str(shared/'evaluation'/second['name']/'status.json')))
        print(json.dumps(dict(event='eval_followup_starting_supplement',workers=second['workers'])),flush=True)
        code = 1
        try:
            # Compare current hosts/checkpoint/runtime against the pilot, not
            # just against the supplement (which has no prior receipts yet).
            run_queue(first,env,dry_run=True,check_runtime=True)
            code = run_queue(second,env)
        finally:
            total = aggregate(summaries(shared,first),summaries(shared,report_second))
            write_json(output/'summary.json',total)
            write_json(output/'status.json',dict(state='completed' if code==0 and
                total['completed']==total['planned'] else 'incomplete',exit_code=code,
                completed=total['completed'],planned=total['planned']))
        return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    return run(args.config,dict(os.environ),dry_run=args.dry_run)


if __name__=='__main__':raise SystemExit(main())

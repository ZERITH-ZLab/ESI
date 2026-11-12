"""Explicit, experiment-scoped exclusions; never alter frozen episode identities."""
import json
import os
from pathlib import Path


def exclusions(shared, experiment):
    # No inherited research exclusions. An operator may explicitly opt into a ledger.
    configured = os.environ.get('ROLLOUT_EVAL_EXCLUSIONS_FILE')
    if not configured:
        return {}
    path = Path(configured)
    data = json.loads(path.read_text())
    if data.get('schema') != 'robodojo.eval_case_exclusions.v1':
        raise ValueError('Invalid evaluation exclusion ledger')
    result = {}
    for row in data['exclusions']:
        if experiment not in row['experiments']:
            continue
        if not row.get('reason') or not row.get('evidence') or row['case_id'] in result:
            raise ValueError('Invalid or duplicate evaluation exclusion')
        result[row['case_id']] = row
    return result


def filter_cases(shared, experiment, cases):
    omitted = exclusions(shared, experiment)
    if set(omitted) - {c['case_id'] for c in cases}:
        raise ValueError('Excluded case missing from evaluation panel')
    return [c for c in cases if c['case_id'] not in omitted]


def unstarted_report(shared, model):
    cases = [dict(case_id=f"{model['task']}__{variant}__g0__l{layout}__pseed0")
             for variant, layout in model['cases']]
    active = filter_cases(shared, model['experiment'], cases)
    omitted = exclusions(shared, model['experiment'])
    return dict(experiment=model['experiment'], original_planned=len(cases), planned=len(active),
                excluded=len(omitted), excluded_cases=list(omitted.values()),
                completed=0, successes=0, failures=0, episodes=[])

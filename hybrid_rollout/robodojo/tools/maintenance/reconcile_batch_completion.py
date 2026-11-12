"""Reconcile a terminal batch whose final bookkeeping was interrupted."""
import argparse
import json
from pathlib import Path
from ...batch import utc
from ...io import write_json


def reconcile(path):
    original = path.read_text()
    batch = json.loads(original)
    rows = batch['episodes']
    if batch['state'] != 'running' or len(rows) != batch['planned_rollouts']:
        raise ValueError('Expected all planned episodes in a stale running batch')
    for row in rows:
        root = Path(row['archive'])
        result = json.loads((root/'controller/result.json').read_text())
        outcome = json.loads((root/'sim/evaluation_outcome.json').read_text())
        artifact = json.loads((root/'artifact_manifest.json').read_text())
        reset = row['reset']
        if not (row['state'] == 'completed' and row['exit_code'] == 0
                and artifact['status'] == 'verified'
                and result.get('complete') is True and outcome.get('complete') is True
                and outcome.get('valid_for_success_rate') is True
                and result['evaluation_case']['case_id'] == row['case_id']
                and outcome['evaluation_case'] == result['evaluation_case']
                and reset.get('all_owned_processes_exited') is True
                and reset.get('ports_released') is True):
            raise ValueError(f'Unverified completion: {root}')
    backup = path.with_name('batch.before_completion_reconciliation.json')
    with backup.open('x') as stream:
        stream.write(original)
    batch.update(state='completed', exit_code=0, finished_utc=rows[-1]['finished_utc'],
        completion_reconciliation=dict(recorded_utc=utc(), backup=str(backup),
            reason='Final bookkeeping failed after every episode completed',
            evidence='Existing per-episode results, artifact checks and cleanup receipts',
            original_process_exit_code_unchanged=True))
    write_json(path, batch)
    print(json.dumps(dict(event='batch_reconciled', batch_path=str(path), episodes=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    reconcile(parser.parse_args().batch)

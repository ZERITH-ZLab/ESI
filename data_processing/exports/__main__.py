import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

from data_processing.subtasks.io import read, write, require
from data_processing.subtasks.pipeline import batch_lock, load_input


def export(run, output, workers=4):
    run, output = Path(run).resolve(), Path(output).resolve()
    batch = run.parent.parent
    require(1 <= workers <= 32, 'workers must be 1..32')
    require(not output.is_relative_to(batch) and not output.is_relative_to(Path(__file__).resolve().parents[2]),
            'Write exports outside annotation inputs and repository')
    manifest = read(batch/'manifest.json')
    require(not manifest['failures'] and not manifest['excluded'], 'Unresolved annotation input selection')
    rows = manifest['episodes']
    require(rows, 'No episodes')
    for row in rows:
        require(not output.is_relative_to(Path(row['archive'])), 'Export overlaps raw input')
        load_input(batch, row)
        annotation = read(run/'episodes'/row['key']/'annotation.json')
        require(annotation['input_hash'] == row['input_hash'], 'Missing/mismatched annotation')
    config = dict(annotation_run=str(run), inputs=[dict(key=r['key'], archive=r['archive'],
                  input_hash=r['input_hash']) for r in rows], action_label='command', allow_wall_time_limit=True)
    with batch_lock(output):
        if (output/'config.json').exists():
            require(read(output/'config.json') == config, 'Export selection changed; use a new directory')
        else:
            write(output/'config.json', config)
        logs = output/'logs'
        logs.mkdir(exist_ok=True)

        def one(item):
            i, row = item
            destination = output/'episodes'/row['key']
            try:
                if not (destination/'manifest.json').exists():
                    staging = output/'staging'/f"{row['key']}_{uuid.uuid4().hex}"
                    staging.parent.mkdir(exist_ok=True)
                    log = logs/f"{staging.name}.log"
                    with log.open('x') as stream:
                        result = subprocess.run([sys.executable, '-u', '-m', 'data_processing.exports',
                            '--extract-one', row['archive'], '--output', str(staging)],
                            stdout=stream, stderr=subprocess.STDOUT)
                    require(result.returncode == 0, f'Extraction failed; inspect {log}')
                    destination.parent.mkdir(exist_ok=True)
                    staging.rename(destination)
                extracted = read(destination/'manifest.json')
                require(extracted['episode_id'] == row['episode_id'] and extracted['steps'] == row['steps']
                        and extracted['source_archive'] == row['archive'], 'Extracted identity mismatch')
                for name, sha in extracted['file_sha256'].items():
                    require(hashlib.sha256((destination/name).read_bytes()).hexdigest() == sha,
                            f'Corrupt export {destination/name}')
                load_input(batch, row)
                result = dict(state='completed', episode_id=row['episode_id'], episode_index=i,
                              steps=row['steps'], sidecar=str(destination.relative_to(output)), mcap=None)
            except Exception as exc:
                result = dict(state='failed', key=row['key'], error=str(exc))
            print(json.dumps(dict(event='command_export_episode', **result)), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, enumerate(rows)))
        failures = [r for r in results if r['state'] == 'failed']
        report = dict(schema='robodojo_command_collection.v1', status='failed' if failures else 'completed',
                      action_label='command', total_episodes=len(rows), total_steps=sum(r.get('steps',0) for r in results),
                      episodes=results, failures=failures, image_storage='original_observation_references',
                      annotation_run=str(run))
        write(output/'report.json', report)
        if not failures:
            write(output/'dataset_collection.json', report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--annotation-run')
    source.add_argument('--extract-one', help=argparse.SUPPRESS)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.extract_one:
        from hybrid_rollout.robodojo.tools.extract_episode import extract
        extract(a.extract_one, a.output, allow_wall_time_limit=True)
    else:
        result = export(a.annotation_run, a.output, a.workers)
        print(json.dumps(dict(event='command_export_finished', episodes=result['total_episodes'],
                              steps=result['total_steps'], failures=result['failures'])), flush=True)
        raise SystemExit(bool(result['failures']))


if __name__ == '__main__':
    main()

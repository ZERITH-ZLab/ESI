"""Independent route, shared auth/client transport and readable batch catalog."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
from pathlib import Path
from queue import Queue
import time

import jsonschema

from data_processing.subtasks.client import CodexClient, APIClient, extract_text
from data_processing.subtasks.io import read, write, digest, require
from data_processing.subtasks.pipeline import batch_lock, load_config, load_input
from data_processing.subtasks.catalog import build_run_catalog
from . import VERSION
from .prepare import model_payload
from .contract import (SCHEMA, AUDIT_SCHEMA, response_schema, validate_candidate,
                       validate_final, merge_patch, ValidationError)

HERE = Path(__file__).parent


def protocol():
    from data_processing import subtasks
    files = list(HERE.glob('*.py')) + [Path(subtasks.__file__).parent / n for n in
                                      ('prepare.py', 'client.py', 'evidence.py', 'catalog.py', 'pipeline.py')]
    return dict(version=VERSION, prompt=(HERE / 'prompt.md').read_text(), schema=SCHEMA,
                audit_schema=AUDIT_SCHEMA, implementation={str(p.relative_to(HERE.parent)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def stats(annotation):
    segments = annotation['segments']
    counts = Counter(s['outcome'] for s in segments)
    return dict(segments=len(segments), **{k: counts[k] for k in ('success','failure','unknown','interrupted')},
                inferred=sum(s['outcome_basis'] == 'inferred' for s in segments),
                needs_review=sum(s['needs_review'] for s in segments),
                training_eligible=sum(s['training_eligible'] for s in segments),
                with_subtask=sum(s['online_conditioning']['subtask'] is not None for s in segments),
                null=sum(s['online_conditioning']['subtask'] is None for s in segments),
                gpt_flagged=sum(s['control_type'] == 'gpt' and
                                (s['needs_review'] or s['outcome'] != 'success') for s in segments))


def validate_saved(prepared, destination):
    expected = validate_final(prepared, read(destination / 'candidate.json'), read(destination / 'audit_final.json'))
    require(expected == read(destination / 'annotation.json'), 'Saved annotation is stale/corrupted; do not silently overwrite')
    return expected


def annotate_one(prepared, destination, cfg, spec, client):
    destination = Path(destination)
    if (destination / 'annotation.json').exists():
        return validate_saved(prepared, destination), 'reused'
    destination.mkdir(parents=True, exist_ok=True)
    payload = model_payload(prepared)
    require(len(json.dumps(payload, ensure_ascii=False)) <= cfg['max_input_chars'],
            'Episode exceeds max_input_chars; increase explicitly, never truncate the ledger')
    # Separate invocation directories retain failed/manual rerun history, including network failures.
    attempts = destination / 'attempts'
    attempts.mkdir(exist_ok=True)
    sequence = max([int(p.name.split('_')[-1]) for p in attempts.glob('attempt_*') if p.is_dir()] or [0]) + 1
    work = attempts / f'attempt_{sequence:04d}'
    work.mkdir()
    write(destination / 'latest_attempt.json', dict(path=str(work)))
    required = payload['required_decision_ids']
    correction, candidate, report, phase = None, None, None, 'initial'
    repairs = 0
    call_index = 0
    while True:
        request = dict(phase=phase, correction=correction)
        write(work / f'request_{call_index:02d}.json', request)
        started = time.time()
        raw = client.complete(spec['prompt'], payload, correction)
        write(work / f'response_{call_index:02d}.json', raw)
        write(work / f'timing_{call_index:02d}.json', dict(started_at=started, ended_at=time.time()))
        output = None
        try:
            output = extract_text(raw, 'codex' if cfg['backend'] == 'codex' else cfg['api_mode'])
            write(work / f'draft_{call_index:02d}.json', output)
            audit = phase == 'audit'
            if audit:
                jsonschema.Draft202012Validator(AUDIT_SCHEMA).validate(output)
            if phase == 'initial':
                proposed = output
            else:
                proposed = merge_patch(candidate, dict(labels=output['labels']), correction['editable_decision_ids'])
            validate_candidate(prepared, proposed)
            if audit:
                proposed_report = dict(reviews=output['reviews'])
                final = validate_final(prepared, proposed, proposed_report)
                candidate, report = proposed, proposed_report
                write(work / f'validation_{call_index:02d}.json', dict(valid=True))
                break
            candidate = proposed
            write(work / f'validation_{call_index:02d}.json', dict(valid=True))
            phase = 'audit'
            correction = dict(mode='audit', previous=candidate, editable_decision_ids=required,
                              errors=[], rejected_output=None,
                              instruction='Audit all decisions; return changed labels plus reviews for every executed decision. Preserve actual inputs and boundaries.')
        except (ValidationError, jsonschema.ValidationError, ValueError, KeyError, TypeError) as error:
            issues = error.issues if isinstance(error, ValidationError) else [dict(
                code='schema_or_parse', decision_id=None, message=str(error))]
            write(work / f'validation_{call_index:02d}.json', dict(valid=False, issues=issues))
            if repairs >= cfg['max_repairs']:
                raise ValueError(f'Annotation failed after {repairs} repair calls: {error}; inspect {work}') from error
            repairs += 1
            # A valid label array with bad references can be repaired locally. A malformed
            # initial response needs a full replacement because there is no safe patch base.
            if phase == 'initial' and not any(i['code'] == 'coverage' for i in issues) and not list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(output)):
                candidate = output
                phase = 'patch'
            editable = sorted({i['decision_id'] for i in issues if i.get('decision_id') in required})
            if phase == 'initial' or phase == 'audit' or not editable or any(i.get('decision_id') is None for i in issues):
                editable = required
            correction = dict(mode=phase, previous=candidate, editable_decision_ids=editable,
                errors=issues, rejected_output=output,
                instruction=('Return a complete replacement labels array.' if phase == 'initial' else
                    'Fix the stated errors against previous. Change only authorized labels. No interval, prompt or control edits. '
                    'For attempt-link errors repair the listed members consistently. Return reviews for all IDs in audit mode.'))
        call_index += 1
    # Commit final labels last. A crash before this file leaves no apparent success.
    write(destination / 'candidate.json', candidate)
    write(destination / 'audit_final.json', report)
    write(destination / 'annotation.json', final)
    write(destination / 'gpt_review.json', dict(segments=[s for s in final['segments']
          if s['control_type'] == 'gpt' and (s['needs_review'] or s['outcome'] != 'success')]))
    return final, 'annotated'


def run_batch(root, cfg, limit=None):
    root = Path(root).resolve()
    require(limit is None or limit > 0, 'limit must be positive')
    with batch_lock(root):
        manifest = read(root / 'manifest.json')
        require(manifest['selection']['version'] == VERSION, 'Use the online route prepare command for this batch')
        spec = protocol()
        identity = {k:v for k,v in cfg.items() if k not in
                    ('workers','timeout_seconds','http_retries','profiles','proxy_file','api_key_env')}
        run_id = digest(dict(protocol=spec, config=identity))[:20]
        run_root = root / 'runs' / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        write(run_root / 'config.json', cfg)
        write(run_root / 'protocol.json', spec)
        write(root / 'latest_run.json', dict(run_id=run_id, run_root=str(run_root)))
        rows = manifest['episodes'][:limit] if limit else manifest['episodes']
        print(json.dumps(dict(event='online_annotation_start', run_root=str(run_root), episodes=len(rows)), ensure_ascii=False), flush=True)
        results, pending = [], []
        for row in rows:
            dest = run_root / 'episodes' / row['key']
            if (dest / 'annotation.json').is_file():
                try:
                    ann = validate_saved(load_input(root, row), dest)
                    results.append(dict(key=row['key'], state='reused', **stats(ann)))
                except Exception as e:
                    results.append(dict(key=row['key'], state='failed', error=str(e)))
            else:
                pending.append(row)
        workers = min(cfg['workers'], len(pending))
        slots = Queue()
        with ExitStack() as stack:
            if workers and cfg['backend'] == 'codex':
                from hybrid_rollout.robodojo.codex_backend.profiles import validate_campaign_sessions, account_lock
                profiles = cfg['profiles'][:workers]
                validate_campaign_sessions(cfg['shared_root'], tuple(profiles))
                for profile in profiles:
                    lease = stack.enter_context(account_lock(profile, cfg['shared_root']))
                    slots.put((profile, lease))
            def worker(row):
                dest = run_root / 'episodes' / row['key']
                slot = None
                started = time.time()
                write(dest / 'status.json', dict(key=row['key'], state='running', started_at=started))
                try:
                    prepared = load_input(root, row)
                    if cfg['backend'] == 'codex':
                        slot = slots.get()
                        client = CodexClient(cfg, *slot, dest / 'codex_runtime', schema_factory=response_schema)
                    else:
                        client = APIClient(cfg, schema_factory=response_schema)
                    ann, state = annotate_one(prepared, dest, cfg, spec, client)
                    result = dict(key=row['key'], state=state, **stats(ann))
                    if result['gpt_flagged']:
                        print(json.dumps(dict(event='gpt_annotation_attention', key=row['key'],
                            flagged=result['gpt_flagged'], report=str(dest / 'gpt_review.json')), ensure_ascii=False), flush=True)
                except Exception as error:
                    result = dict(key=row['key'], state='failed', error=f'{type(error).__name__}: {error}')
                finally:
                    if slot is not None:
                        slots.put(slot)
                result.update(started_at=started, ended_at=time.time(), directory=str(dest))
                write(dest / 'status.json', result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return result
            if pending:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results.extend(executor.map(worker, pending))
        # Entire-batch index excludes stale/failed episodes, even with --limit or append.
        index = []
        for row in manifest['episodes']:
            dest = run_root / 'episodes' / row['key']
            if not (dest / 'annotation.json').is_file():
                continue
            try:
                ann = validate_saved(load_input(root, row), dest)
                index += [dict(episode_key=row['key'], annotation=str(dest / 'annotation.json'), **s)
                          for s in ann['segments']]
            except (ValueError, OSError, KeyError, jsonschema.ValidationError):
                continue
        write(run_root / 'segments.json', index)
        catalog = build_run_catalog(root, run_root, manifest)
        summary = dict(run_id=run_id, run_root=str(run_root), named_run_root=catalog['named_run_root'],
                       results=results, failures=sum(r['state']=='failed' for r in results))
        write(run_root / 'summary.json', summary)
        return summary


def validate_run(root, run_id=None):
    root = Path(root).resolve()
    run_id = run_id or read(root / 'latest_run.json')['run_id']
    require(Path(run_id).name == run_id and run_id not in ('.','..'), 'Invalid run ID')
    results = []
    with batch_lock(root):
        for row in read(root / 'manifest.json')['episodes']:
            try:
                ann = validate_saved(load_input(root, row), root / 'runs' / run_id / 'episodes' / row['key'])
                results.append(dict(key=row['key'], state='valid', **stats(ann)))
            except (ValueError, OSError, KeyError, jsonschema.ValidationError) as error:
                results.append(dict(key=row['key'], state='failed', error=str(error)))
    return dict(results=results, failures=sum(r['state']=='failed' for r in results))

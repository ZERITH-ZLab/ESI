"""Frozen 20-scene digit collection protocol, separate from official evaluation."""
import argparse
import json
from pathlib import Path

from .io import require

SCHEMA = 'hybrid_rollout.robodojo.collection_panel.v1'
PROTOCOL = dict(task='arrange_largest_number', eval_seed=0, repeats=5,
                order='repeat_major', native_step_limit=1050, step_limit=1500,
                max_seconds=10800, policy_rng_seed=0)


def protocol(repeats=5):
    require(type(repeats) is int and repeats > 0, 'Collection repeats must be a positive integer')
    return dict(PROTOCOL, repeats=repeats)


def sampling_job_index(panel, case_id, repeat_id):
    """Stable task-panel identity, independent of dispatch/subset order."""
    require(type(repeat_id) is int and 1 <= repeat_id <= panel['collection']['repeats'],
            'Collection repeat out of range')
    ids = [c['case_id'] for c in panel['cases']]
    require(case_id in ids, 'Collection case not in task panel')
    return (repeat_id - 1) * len(ids) + ids.index(case_id)


def specs():
    for i, (variant, layout) in enumerate(
            [(v, n) for v, count in (('standard', 15), ('random', 5)) for n in range(count)]):
        task = PROTOCOL['task']
        yield dict(case_id=f'{task}__{variant}__g0__l{layout}', task=task,
                   runtime_task=task + ('_random' if variant == 'random' else ''),
                   variant=variant, replica_id=3, rollout_index=i, eval_seed=0,
                   layout_id=layout, reset_seed=layout, simulator_initial_seed=0, policy_rng_seed=0)


def validate(panel):
    if panel.get('collection', {}).get('kind') == 'task20_v1':
        from .collection_suite import validate_task_panel
        return validate_task_panel(panel)
    from .evaluation import canonical_sha256
    require(panel.get('schema') == SCHEMA and panel.get('split') == 'train', 'Not a collection panel')
    require(panel.get('collection') == protocol(panel.get('collection', {}).get('repeats'))
            and panel.get('eval_seed') == 0, 'Collection protocol changed')
    require(panel['panel_sha256'] == canonical_sha256({k:v for k,v in panel.items() if k != 'panel_sha256'}),
            'Collection panel hash changed')
    require(len(panel['cases']) == 20, 'Collection requires 15 standard + 5 random cases')
    for actual, expected in zip(panel['cases'], specs()):
        require(all(actual.get(k) == v for k,v in expected.items()), 'Collection case ordering/identity changed')
        require(actual['native_support_trajectories'] == [], 'Unexpected support trajectories')
        path = Path(actual['layout']['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe collection layout path')
    return panel


def freeze(source, repeats=5):
    from .evaluation import freeze_panel, layout_files, file_record, canonical_sha256
    source = Path(source).resolve()
    panel = freeze_panel(source, 'arrange_largest_number_collection20_g0_v1', 0)
    panel.update(schema=SCHEMA, scope='training_collection_not_official_evaluation', split='train',
                 collection=protocol(repeats), selection='Fixed before collection: standard 0..14, random 0..4', cases=[])
    for spec in specs():
        paths = layout_files(source, spec['runtime_task'], 0)
        require(spec['layout_id'] < len(paths), f'Missing native layout: {spec["case_id"]}')
        panel['cases'].append(dict(spec, layout=file_record(source, paths[spec['layout_id']]),
                                  native_support_trajectories=[]))
    panel['panel_sha256'] = canonical_sha256({k:v for k,v in panel.items() if k != 'panel_sha256'})
    return validate(panel)


def runtime_settings(env, panel):
    """No budget override leaks into legacy evaluation launches."""
    requested = int(env.get('ROLLOUT_COLLECTION_STEP_LIMIT', '0'))
    if panel.get('schema') != SCHEMA:
        require(requested == 0, 'Collection step override requires a collection panel')
        return {}
    validate(panel)
    limits = panel['collection']
    require(requested == limits['step_limit'], 'Collection step limit differs from frozen panel')
    require(int(env.get('ROLLOUT_MAX_SECONDS', '0')) == limits['max_seconds'], 'Collection wall limit differs from frozen panel')
    require(env.get('ROLLOUT_ACCOUNT_POOL_FILE'), 'Repeated collection requires the account pool')
    require(env.get('ROLLOUT_CONTROLLER_ROUTE') in ('auto_infer_v1', 'subtask_feedforward_v1')
            and env.get('ROLLOUT_EVALUATION_METHOD') == 'pi05_plus_gpt'
            and env.get('ROLLOUT_ENABLE_SCENE_QUERY') == '1'
            and env.get('ROLLOUT_ENABLE_MOTION_TOOLS') == '1', 'Collection requires auto-infer or feedforward with query/motion tools')
    if panel['collection'].get('kind') == 'task20_v1':
        # The multi-task release records actual teacher settings in resume identity.
        # The old single-task protocol below retains its explicit historical contract.
        require(bool(env.get('ROLLOUT_MODEL','').strip())
                and env.get('ROLLOUT_REASONING_EFFORT') in ('low','medium','high','xhigh','max','ultra')
                and int(env.get('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT','0')) > 0,
                'Invalid collection teacher configuration')
    else:
        require(env.get('ROLLOUT_MODEL', 'gpt-6-astra') == 'gpt-6-astra'
                and env.get('ROLLOUT_REASONING_EFFORT', 'xhigh') == 'xhigh'
                and int(env.get('ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT', '96000')) == 96000,
                'Collection teacher configuration differs')
    validate(panel)
    return dict(collection_protocol=panel['collection'].copy())


def main():
    from .evaluation import read_panel, verify_assets
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    panel = read_panel(args.output) if args.output.exists() else freeze(args.source, args.repeats)
    validate(panel)
    require(panel['collection'] == protocol(args.repeats),
            'Existing manifest has different repeats; use a separate manifest')
    verify_assets(panel, args.source, verify_source=False)
    if not args.dry_run and not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never overwrite an existing collection identity.
        with args.output.open('x') as stream:
            json.dump(panel, stream, indent=2)
            stream.write('\n')
    print(json.dumps(dict(event='collection_plan', manifest=str(args.output), cases=20,
                         planned_episodes=len(panel['cases']) * args.repeats,
                         **panel['collection']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

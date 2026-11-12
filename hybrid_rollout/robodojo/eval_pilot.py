"""Prepare/run the ten-task round-3 strict evaluation pilot; never collect train data."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .collection_suite import RANDOM_TASKS
from .downsampling import SelectionConfig
from .eval_sequence import read_config, run_sequence

TASKS = ('pack_objects_into_box', 'arrange_largest_number', 'sort_nesting_dolls_by_size',
         'classify_objects', 'fill_pen_holder', 'organize_table', 'imitate_sorting_sequence',
         'stack_blocks_by_language', 'classify_objects_by_language', 'sweep_blocks')


def make_config(name, checkpoint, pool, selection, epsilons, workers):
    models = []
    for epsilon in epsilons:
        label = format(epsilon, '.6g').replace('.', 'p').replace('-', 'm')
        config = asdict(SelectionConfig(gripper_epsilon=epsilon))
        for task in TASKS:
            if task in RANDOM_TASKS and selection != 'standard2':
                count = 2 if selection == 'each_variant2' else 1
                cases = [[v, i] for v in ('standard', 'random') for i in range(count)]
            else:
                cases = [['standard', 0], ['standard', 1]]
            models.append(dict(experiment=f'{name}__eps{label}__{task}', task=task,
                cases=cases, checkpoint=str(checkpoint), policy_input='task_subtask',
                downsampling=dict(mode='error', config=config)))
    return dict(schema='robodojo.eval_sequence.v1', name=name, workers=workers,
                account_pool_file=str(pool), models=models)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--account-pool', default='/root/.config/gpt-as-policy/account_pool.json')
    p.add_argument('--case-selection', choices=['one_per_variant', 'standard2', 'each_variant2'], required=True)
    p.add_argument('--gripper-epsilon', type=float, nargs='+', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--config-output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = make_config(args.name, args.checkpoint, args.account_pool,
                      args.case_selection, args.gripper_epsilon, args.workers)
    # Config must never overwrite a different already prepared experiment.
    if args.config_output.exists():
        if json.loads(args.config_output.read_text()) != cfg:
            raise ValueError('Existing config differs; use a new config path/name')
    else:
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        with args.config_output.open('x') as stream:
            json.dump(cfg, stream, ensure_ascii=False, indent=2)
    cfg = read_config(args.config_output)
    print(json.dumps(dict(event='eval_pilot_config', path=str(args.config_output.resolve()),
        episodes=sum(len(m['cases']) for m in cfg['models']), workers=cfg['workers'],
        scheduling='tasks sequential; cases within each task parallel',
        automatic_force_hold=False, gpt_closure_wait=True)), flush=True)
    if args.prepare_only:
        return 0
    return run_sequence(cfg, dict(os.environ), dry_run=args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())

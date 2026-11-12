"""Frozen native-budget task subsets using the existing strict eval executor."""
from copy import deepcopy
from pathlib import Path

from .collection_suite import TASKS, task_specs, freeze_task, native_limit
from .evaluation import canonical_sha256, freeze_panel, layout_files, file_record
from .io import require

KIND = 'task_subset_v1'
# Evaluation-only inventory: do not append unseen tasks to the training suite.
TRANSFER_TASKS = {'solve_equation': 300}
TRANSFER_SCOPE = 'zero_flywheel_shot_task_transfer_not_pretraining_unseen'


def available_specs(task):
    if task not in TRANSFER_TASKS:
        return list(task_specs(task))
    return [dict(case_id=f'{task}__standard__g0__l{i}', task=task,
        runtime_task=task, variant='standard', replica_id=3, rollout_index=i,
        eval_seed=0, layout_id=i, reset_seed=i, simulator_initial_seed=0,
        policy_rng_seed=0) for i in range(55)]


def selected_specs(task, selection):
    require(task in TASKS or task in TRANSFER_TASKS, 'Unsupported eval task')
    require(isinstance(selection, list) and bool(selection), 'Empty eval case selection')
    available = {(c['variant'], c['layout_id']): c for c in available_specs(task)}
    keys = [tuple(item) for item in selection]
    require(len(set(keys)) == len(keys) and all(k in available for k in keys), 'Invalid eval case selection')
    return [dict(available[key], replica_id=3, rollout_index=i,
                 evaluation_repeat=1, case_id=available[key]['case_id']+'__pseed0')
            for i, key in enumerate(keys)]


def validate_task_eval(panel):
    from .regression_eval import SCHEMA, VERSION
    require(panel.get('schema') == SCHEMA and panel.get('protocol') == VERSION, 'Invalid task eval protocol')
    require(panel.get('panel_sha256') == canonical_sha256(
        {k:v for k,v in panel.items() if k != 'panel_sha256'}), 'Eval panel hash mismatch')
    require(panel.get('queue_mode') == 'all_policy_seeds_v1' and panel.get('policy_rng_seed') is None,
            'Task eval uses per-case seeds')
    require(type(panel.get('native_step_limit')) is int and panel['native_step_limit'] > 0,
            'Invalid native budget')
    if panel['task'] in TRANSFER_TASKS:
        require(panel.get('scope') == TRANSFER_SCOPE and
                panel['native_step_limit'] == TRANSFER_TASKS[panel['task']],
                'Transfer scope/native budget changed')
    expected = selected_specs(panel['task'], panel['case_selection'])
    require(len(panel['cases']) == len(expected), 'Eval case count changed')
    for case, spec in zip(panel['cases'], expected):
        require(all(case.get(k) == v for k,v in spec.items()), 'Eval case identity changed')
        support = case.get('native_support_trajectories', [])
        paths = ([f'Assets/Traj/RoboDojo/imitate_sorting_sequence/0/{case["layout_id"]}.pkl']
                 if panel['task'] == 'imitate_sorting_sequence' else [])
        require([r['path'] for r in support] == paths, 'Eval support mapping changed')
        for record in [case['layout'], *support]:
            p = Path(record['path'])
            require(not p.is_absolute() and '..' not in p.parts, 'Unsafe eval asset path')
    return panel


def freeze(source, task, selection):
    from .regression_eval import SCHEMA, VERSION
    if task in TRANSFER_TASKS:
        source = Path(source).resolve()
        parent = freeze_panel(source, 'transfer_inventory', 0)
        limit = native_limit(source, task)
        require(limit == TRANSFER_TASKS[task], 'Transfer native budget changed')
        require(not layout_files(source, task+'_random', 0), 'Transfer variant inventory changed')
        parent['collection'] = dict(native_step_limit=limit)
        paths = layout_files(source, task, 0)
        chosen = selected_specs(task, selection)
        require(all(c['layout_id'] < len(paths) for c in chosen), 'Missing transfer layouts')
        require(all(paths[c['layout_id']].name == f'{task}_{c["layout_id"]}.json' for c in chosen),
                'Native layout index does not match requested file number')
        parent['cases'] = [dict(spec, layout=file_record(source, paths[spec['layout_id']]),
                                native_support_trajectories=[]) for spec in chosen]
    else:
        parent = freeze_task(source, task, [1, 1])
    lookup = {(c['variant'], c['layout_id']): c for c in parent['cases']}
    panel = {k: deepcopy(parent[k]) for k in ('native_source_commit', 'native_source_files')}
    panel.update(schema=SCHEMA, protocol=VERSION, task_eval_kind=KIND,
        panel_id=f'{task}_native_subset_v1', task=task, case_selection=selection,
        native_step_limit=parent['collection']['native_step_limit'],
        scope=TRANSFER_SCOPE if task in TRANSFER_TASKS else 'training_layout_regression_not_heldout', policy_rng_seed=None, eval_seed=0,
        queue_mode='all_policy_seeds_v1', cases=[])
    for spec in selected_specs(task, selection):
        original = lookup[spec['variant'], spec['layout_id']]
        panel['cases'].append(dict(spec, layout=original['layout'],
            native_support_trajectories=original['native_support_trajectories']))
    panel['panel_sha256'] = canonical_sha256(panel)
    return validate_task_eval(panel)

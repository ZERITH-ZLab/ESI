"""Explicit single-task standard panels, isolated from the legacy 60-case panel."""
from pathlib import Path
from ..evaluation import canonical_sha256, freeze_panel, layout_files, file_record
from ..io import require

SCHEMA = 'hybrid_rollout.robodojo.single_task_panel.v1'


def validate_task_panel(panel):
    """Validate all task, group, split and reset identities before execution."""
    require(panel['panel_sha256'] == canonical_sha256({k:v for k,v in panel.items() if k!='panel_sha256'}), 'Panel hash changed')
    require(panel['schema']==SCHEMA and panel['split'] in ('train','validation','test'), 'Unknown task panel')
    cases=panel['cases']
    require(1<=len(cases)<=60 and len({c['layout_id'] for c in cases})==len(cases), 'Duplicate or empty cases')
    for i,c in enumerate(cases):
        layout=c['layout_id']; group=panel['eval_seed']
        require(type(layout) is int and 0<=layout<60 and group in (0,1,2), 'Invalid published layout')
        expected=dict(task='deposit_coin',runtime_task='deposit_coin',variant='standard',
            replica_id=0,rollout_index=i,eval_seed=group,layout_id=layout,reset_seed=layout,
            simulator_initial_seed=0,policy_rng_seed=0,
            case_id=f'deposit_coin__standard__g{group}__l{layout}')
        require(all(c.get(k)==v for k,v in expected.items()), 'Case semantics changed')
        for record in [c['layout'],*c['native_support_trajectories']]:
            require(not Path(record['path']).is_absolute() and '..' not in Path(record['path']).parts, 'Unsafe layout reference')
    return panel


def freeze_task_panel(source, panel_id, group, layouts, split):
    """Reuse native source inventory and freeze explicitly selected layouts."""
    source=Path(source)
    panel=freeze_panel(source,panel_id,group)
    files=layout_files(source,'deposit_coin',group)
    panel.update(schema=SCHEMA,scope='single_task_standard_flywheel',split=split,cases=[])
    for i,layout in enumerate(layouts):
        panel['cases'].append(dict(case_id=f'deposit_coin__standard__g{group}__l{layout}',
            task='deposit_coin',runtime_task='deposit_coin',variant='standard',replica_id=0,
            rollout_index=i,eval_seed=group,layout_id=layout,reset_seed=layout,
            simulator_initial_seed=0,policy_rng_seed=0,
            layout=file_record(source,files[layout]),native_support_trajectories=[]))
    panel['panel_sha256']=canonical_sha256({k:v for k,v in panel.items() if k!='panel_sha256'})
    return validate_task_panel(panel)

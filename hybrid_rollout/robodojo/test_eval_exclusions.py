import json
from pathlib import Path

import pytest

from .eval_exclusions import exclusions, filter_cases, unstarted_report
from .io import write_json


def ledger(root, cases=('bad',), experiments=('trained','base')):
    path=root/'evaluation/case_exclusions.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,dict(schema='robodojo.eval_case_exclusions.v1',exclusions=[
        dict(case_id=c,experiments=list(experiments),reason='invalid_native_layout',evidence=['attempts']) for c in cases]))


def test_scoped_exclusions_do_not_change_other_experiments(tmp_path,monkeypatch):
    ledger(tmp_path)
    monkeypatch.setenv("ROLLOUT_EVAL_EXCLUSIONS_FILE",str(tmp_path/"evaluation/case_exclusions.json"))
    cases=[dict(case_id='bad'),dict(case_id='good')]
    assert filter_cases(tmp_path,'trained',cases)==[cases[1]]
    assert filter_cases(tmp_path,'base',cases)==[cases[1]]
    assert filter_cases(tmp_path,'other',cases)==cases
    with pytest.raises(ValueError,match='missing'):filter_cases(tmp_path,'base',[])


def test_report_keeps_exclusion_audit_but_removes_from_denominator(tmp_path,monkeypatch):
    from . import regression_eval as e
    ledger(tmp_path)
    monkeypatch.setenv("ROLLOUT_EVAL_EXCLUSIONS_FILE",str(tmp_path/"evaluation/case_exclusions.json"))
    path=tmp_path/'eval_panels/trained/queue.json';path.parent.mkdir(parents=True);path.write_text('{}')
    cases=[dict(case_id=c,policy_rng_seed=0,variant='standard',layout_id=i) for i,c in enumerate(['bad','good'])]
    monkeypatch.setattr(e,'read_panel',lambda _:dict(cases=cases))
    batch=tmp_path/'results/trained/_replicas/replica_3/attempt_0/batch.json';batch.parent.mkdir(parents=True)
    # Even a successful receipt for an excluded case must not enter the comparison.
    write_json(batch,dict(episodes=[dict(case_id=c['case_id'],archive='old',artifact_status='verified',
        result=dict(complete=True),evaluation_outcome=dict(complete=True,valid_for_success_rate=True,native_success=True),
        reset=dict(all_owned_processes_exited=True,ports_released=True)) for c in cases]))
    r=e.report(tmp_path,'trained',unified=True)
    assert (r['original_planned'],r['planned'],r['excluded'],r['completed'],r['successes'])==(2,1,1,1,1)
    assert r['episodes'][0]['state']=='excluded' and r['episodes'][0]['attempt_archives']==['old']


def test_dispatch_filters_before_job_allocation_and_preserves_identity(tmp_path,monkeypatch):
    from . import eval_queue as q, dual_campaign
    ledger(tmp_path)
    monkeypatch.setenv("ROLLOUT_EVAL_EXCLUSIONS_FILE",str(tmp_path/"evaluation/case_exclusions.json"))
    env=dict(ROLLOUT_SHARED_ROOT=str(tmp_path),ROLLOUT_RESUME_IDENTITY='unchanged')
    monkeypatch.setattr(q,'campaign_env',lambda *a:dict(env))
    monkeypatch.setattr(dual_campaign,'configure',lambda e:(e,[dict(case_id='bad'),dict(case_id='good')]))
    campaigns=q.prepare_campaigns(dict(workers=8),{},[(dict(experiment='trained'),{},'panel')],tmp_path)
    assert campaigns[0][1]==[dict(case_id='good')]
    assert campaigns[0][0]['ROLLOUT_RESUME_IDENTITY']=='unchanged'


def test_historical_exclusions_are_not_inherited(tmp_path,monkeypatch):
    monkeypatch.delenv("ROLLOUT_EVAL_EXCLUSIONS_FILE",raising=False)
    root=Path(__file__).parent
    configs=[json.loads((root/f).read_text()) for f in ['eval_queue.round3_pilot20.json',
        'eval_queue.round3_remaining180.json','eval_queue.baseline_full200.json']]
    task='imitate_sorting_sequence'
    names=[m['experiment'] for cfg in configs[1:] for m in cfg['models'] if m['task']==task]
    ledger(tmp_path,[f'{task}__standard__g0__l{i}__pseed0' for i in [8,17]],names)
    counts=[sum(unstarted_report(tmp_path,m)['planned'] for m in cfg['models']) for cfg in configs]
    assert counts==[20,180,200]


def test_excluded_cases_do_not_block_retry(tmp_path):
    from .test_eval_retry import fixture
    from .eval_retry import assess_queue
    cfg,reports=fixture(tmp_path)
    reports[0]['episodes'][0]['state']='excluded'
    assert assess_queue(tmp_path,cfg,reports)==dict(retryable=True,reasons={},blocked=[])

import hashlib
import json
from copy import deepcopy

import pytest
from .io import write_json
from .resume_upgrade import prepare, accepted_identities, compatible
from .account_pool import completion_receipt, collection_completed
from .test_collection import setup


def test_explicit_upgrade_and_future_resume(tmp_path):
    _, _, env = setup(tmp_path)
    old = dict(source_sha256='old', model='astra', effort='xhigh', checkpoint_sha256='weights')
    new = dict(old, source_sha256='new')
    base = tmp_path/'results/test'
    replica = base/'_replicas/replica_3'
    archive = base/'sample'
    (archive/'dual_machine').mkdir(parents=True)
    (archive/'dual_machine/remote_cleanup_status.txt').write_text('0')
    reset = dict(all_owned_processes_exited=True, ports_released=True)
    row = dict(case_id='case', repeat_id=1, archive=str(archive), reset=reset,
               state='completed', artifact_status='verified', result=dict(complete=True),
               evaluation_outcome=dict(complete=True))
    (replica/'attempt_0').mkdir(parents=True)
    receipt = replica/'attempt_0/batch.json'
    write_json(receipt, dict(resume_identity=old, final_reset=reset, episodes=[row]))
    before = receipt.read_bytes()
    env.update(ROLLOUT_RESUME_IDENTITY=json.dumps(new), ROLLOUT_RUN_MODE='resume')
    assert collection_completed(replica, new, 'case', 1) is None
    prepare(env)
    assert accepted_identities(base, new) == [new]
    prepare(dict(env, ROLLOUT_ALLOW_SOURCE_UPGRADE='1'))
    assert collection_completed(replica, new, 'case', 1) == row
    assert receipt.read_bytes() == before
    assert accepted_identities(base, dict(new, source_sha256='unapproved')) == [dict(new, source_sha256='unapproved')]
    for key in ('model', 'effort', 'checkpoint_sha256'):
        assert not compatible(old, dict(new, **{key:'changed'}))
    with pytest.raises(ValueError, match='more than source'):
        prepare(dict(env, ROLLOUT_ALLOW_SOURCE_UPGRADE='1',
                     ROLLOUT_RESUME_IDENTITY=json.dumps(dict(new, model='other'))))


def test_late_outcome_requires_hashed_matching_evidence(tmp_path):
    identity = dict(source_sha256='old')
    case = dict(case_id='case')
    result = dict(episode_id='episode', step_id=10, complete=True, evaluation_case=case)
    outcome = dict(complete=True, native_control_steps=10, evaluation_case=case,
                   valid_for_success_rate=True, native_success=False)
    files = []
    for name, value in [('sim/evaluation_outcome.json', outcome), ('controller/result.json', result),
                        ('resume_identity.json', identity), ('collection_job.json', dict(case_id='case', repeat_id=1))]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
        files.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(tmp_path/'artifact_manifest.json', dict(status='verified', episode_id='episode', all_immutable_artifacts=files))
    row = dict(state='completed', artifact_status='verified', archive=str(tmp_path),
               case_id='case', repeat_id=1, result=result, evaluation_outcome=None)
    original = deepcopy(row)
    assert completion_receipt(row, identity)['evaluation_outcome'] == outcome
    assert row == original
    assert completion_receipt(dict(row, repeat_id=2), identity)['evaluation_outcome'] is None
    assert completion_receipt(dict(row, state='failed'), identity)['evaluation_outcome'] is None
    write_json(tmp_path/'sim/evaluation_outcome.json', dict(outcome, native_success=True))
    assert completion_receipt(row, identity)['evaluation_outcome'] is None

import copy
from pathlib import Path

import pytest

from .evaluation import case_identity
from .io import sha256, write_json
from .paired_evaluation import comparison
from .phase_gate import verify_certificate
from .test_evaluation import fake_panel_file


def certificate(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    chosen = [c for c in panel['cases'] if c['rollout_index'] != 2
              ]  # Ten tasks, five cases each; test identity fixture only.
    rows = []
    for i, case in enumerate(chosen):
        archive = tmp_path/'artifacts'/str(i)
        (archive/'controller').mkdir(parents=True)
        (archive/'sim').mkdir()
        expected = case_identity(panel, case)
        write_json(archive/'controller/result.json', dict(complete=True, evaluation_case=expected))
        write_json(archive/'sim/evaluation_outcome.json', dict(complete=True, valid_for_success_rate=True,
            evaluation_case=expected, native_success=i < 8))
        write_json(archive/'artifact_manifest.json', dict(status='verified'))
        rows.append(dict(case_id=case['case_id'], task=case['task'], evaluation_case=expected,
            native_success=i < 8, archive=str(archive), reused_v1_success=i < 8,
            context_version='v1' if i < 8 else 'v2',
            result_sha256=sha256(archive/'controller/result.json'),
            outcome_sha256=sha256(archive/'sim/evaluation_outcome.json'),
            artifact_manifest_sha256=sha256(archive/'artifact_manifest.json')))
    selection = dict(method='pi05_plus_gpt', complete=True, platform_verified=True,
                     cases=rows, native_successes=8)
    document = dict(schema='robodojo.hybrid50_completion.v1', panel_sha256=panel['panel_sha256'], selection=selection)
    path = tmp_path/'certificate.json'; write_json(path, document)
    return path, panel, document


def test_phase_gate_all50_native_success_or_failure(tmp_path):
    path, panel, doc = certificate(tmp_path)
    assert verify_certificate(path, panel) == doc
    assert sum(r['native_success'] for r in doc['selection']['cases']) == 8


@pytest.mark.parametrize('change', ['missing', 'only49', 'duplicate', 'unverified', 'seed',
                                  'result_changed', 'v1_failure', 'nonexempt_v1'])
def test_phase_gate_rejects_invalid_or_incomplete(tmp_path, change):
    path, panel, doc = certificate(tmp_path)
    rows = doc['selection']['cases']
    if change == 'missing':
        path = tmp_path/'notyet.json'
    elif change == 'only49':
        rows.pop()
    elif change == 'duplicate':
        rows[1] = rows[0]
    elif change == 'unverified':
        doc['selection']['platform_verified'] = False
    elif change == 'seed':
        rows[0]['evaluation_case']['reset_seed'] = 123
    elif change == 'result_changed':
        write_json(Path(rows[0]['archive'])/'controller/result.json', dict(complete=False))
    elif change == 'v1_failure':
        rows[9].update(reused_v1_success=True, context_version='v1')
    elif change == 'nonexempt_v1':
        rows[9]['context_version'] = 'v1'
    if change != 'missing':
        write_json(path, doc)
    with pytest.raises(ValueError):
        verify_certificate(path, panel)


def test_paired_summary_checks_seed_and_labels_v1(tmp_path):
    _, _, doc = certificate(tmp_path)
    hybrid = doc['selection']; direct = copy.deepcopy(hybrid)
    direct['method'] = 'gpt_only'
    report = comparison(hybrid, direct)
    assert report['hybrid_success_rate'] == 8/50
    assert report['reused_v1_success_count'] == 8 and len(report['per_task']) == 10
    direct['cases'][0]['evaluation_case']['reset_seed'] = 345
    with pytest.raises(ValueError, match='aligned'):
        comparison(hybrid, direct)

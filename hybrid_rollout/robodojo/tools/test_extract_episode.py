import json

import numpy as np
import pytest

from .extract_episode import CAMERAS, decision_index, extract


def fixture(tmp_path):
    root = tmp_path/'raw'
    sim = root/'sim/e'
    (sim/'observations').mkdir(parents=True)
    (root/'controller').mkdir()
    def save(p, value):
        p.write_text(json.dumps(value))
    save(root/'sim/summary.json', dict(episode_id='e', step_id=2, control_dt=.04, reason='terminal'))
    save(root/'sim/evaluation_outcome.json', dict(native_success=True))
    states = np.asarray([np.full(14, t/10, np.float32) for t in range(3)])
    for t in range(3):
        np.savez(sim/'observations'/f'{t:06d}.npz', states=states[t], instruction='task',
                 eef_positions=np.zeros((2, 3)), eef_quaternions_wxyz=np.tile([1, 0, 0, 0], (2, 1)),
                 **{k: np.zeros((2, 3, 3), np.uint8) for k in CAMERAS})
    commands = states[1:]+.05
    for t in range(2):
        save(sim/f'action_{t:06d}.json', dict(valid=True, step_id=t+1,
             executed_action=commands[t].tolist(), obs=dict(states=states[t+1].tolist()),
             terminated=t == 1, truncated=False, success=t == 1))
    history = []
    for d, (a, b) in enumerate(((0, 1), (1, 1), (1, 2))):
        history.append(dict(decision=d, start_tick=a, end_tick=b, executed_steps=b-a,
                            response=dict(mode='student' if d == 0 else 'eef',
                                          **({'motion': {}} if d else {}))))
        np.savez(root/'controller'/f'execution_{d:03d}.npz', actions=commands[a:b], states=states[a+1:b+1])
    save(root/'controller/history.json', history)
    return root


def test_alignment_and_labels(tmp_path):
    raw = fixture(tmp_path)
    out = tmp_path/'out'
    report = extract(raw, out)
    assert report['status'] == 'passed'
    assert report['zero_step_decisions'] == 1
    with np.load(out/'steps.npz') as data:
        np.testing.assert_allclose(data['state'][0], 0)
        np.testing.assert_allclose(data['next_state'][0], .1)
        np.testing.assert_allclose(data['command'][0], .15)
        assert list(data['decision_id']) == [0, 2]
    assert json.loads((out/'manifest.json').read_text())['training_action_label'] is None
    with pytest.raises(FileExistsError): extract(raw, out)


def test_corrupt_receipt_fails_without_manifest(tmp_path):
    raw = fixture(tmp_path)
    p = raw/'sim/e/action_000000.json'
    row = json.loads(p.read_text()); row['obs']['states'] = [0]*14
    p.write_text(json.dumps(row))
    with pytest.raises(AssertionError): extract(raw, tmp_path/'out')
    assert not (tmp_path/'out/manifest.json').exists()
    assert json.loads((tmp_path/'out/report.json').read_text())['status'] == 'failed'


def test_history_gap_rejected():
    with pytest.raises(ValueError):
        decision_index([dict(decision=0, start_tick=1, end_tick=2, executed_steps=1)], 2)


def test_missing_observation_rejected(tmp_path):
    raw = fixture(tmp_path)
    (raw/'sim/e/observations/000001.npz').unlink()
    with pytest.raises(ValueError, match='Observation count'):
        extract(raw, tmp_path/'out')
    assert not (tmp_path/'out/manifest.json').exists()


def test_controller_command_mismatch_rejected(tmp_path):
    raw = fixture(tmp_path)
    p = raw/'controller/execution_000.npz'
    with np.load(p) as data:
        states = data['states'].copy()
    np.savez(p, actions=np.zeros((1, 14)), states=states)
    with pytest.raises(AssertionError): extract(raw, tmp_path/'out')
    assert not (tmp_path/'out/manifest.json').exists()

import hashlib
import json
from types import SimpleNamespace

import pytest

from . import task_memory as m
from .test_memory import row
from ..io import write_json
from ..test_network_continue import Transport, event, completed


def env(tmp_path):
    return dict(ROLLOUT_ENABLE_TASK_MEMORY='1', ROLLOUT_CONTROLLER_ROUTE='auto_infer_v1',
                ROLLOUT_EVALUATION_METHOD='pi05_plus_gpt', ROLLOUT_SHARED_ROOT=str(tmp_path),
                ROLLOUT_EXPERIMENT_ID='test', ROLLOUT_RESUME_IDENTITY='{"source": "same"}')


def test_focus_uses_executed_steps_not_model_claims():
    rows = [row() for _ in range(8)]
    for i, r in enumerate(rows):
        r['decision'] = i
    rows[1]['response']['mode'] = 'eef'; rows[1]['executed_steps'] = 3
    rows[3]['response']['mode'] = 'edit'; rows[3]['executed_steps'] = 5
    rows[4]['response']['mode'] = 'eef'; rows[4]['executed_steps'] = 40
    packet = m.focus_packet(rows)
    assert packet['focus_decisions'] == [3, 4]
    assert packet['correction_steps'] == 45
    assert packet['detailed_decisions'] == [2, 3, 4, 5]
    assert 'D0:' in packet['timeline'] and 'D7:' in packet['timeline']
    assert m.focus_packet([row()])['focus_decisions'] == []


def test_long_focus_is_bounded_and_memory_changes_resume_identity(tmp_path):
    from ..dual_campaign import controller_route_identity
    rows = [row() for _ in range(80)]
    for i, r in enumerate(rows):
        r['decision'] = i
        r['response']['mode'] = 'eef'
    packet = m.focus_packet(rows)
    assert len(packet['detailed_decisions']) == 20
    assert packet['omitted_detail_count'] == 60
    assert packet['detailed_decisions'][0] == 0 and packet['detailed_decisions'][-1] == 79
    e = env(tmp_path)
    assert controller_route_identity(e)['task_memory']['max_words'] == 999
    assert 'task_memory' not in controller_route_identity(dict(e, ROLLOUT_ENABLE_TASK_MEMORY='0'))


def test_inject_valid_same_experiment_and_identity(tmp_path):
    e = env(tmp_path)
    for i in range(9):
        archive = tmp_path/f'results/test/arrange_largest_number/standard/eval_seed_0/layout_{i}/replica_3/repeat_001/attempt_000'
        memory = archive/'controller/memory'; memory.mkdir(parents=True)
        body = f'Experience {i}: no unsupported theory.' if i != 7 else ''
        (memory/'episode_summary.md').write_text(body)
        write_json(memory/'episode_summary.json', dict(version=m.VERSION, status='completed',
            max_correction_streak=60, correction_ratio=i/10, correction_outcome='failed_or_unverified',
            base_task='arrange_largest_number', finished_at=i,
            sha256=hashlib.sha256(body.encode()).hexdigest()))
        write_json(archive/'resume_identity.json', dict(source='different' if i == 8 else 'same'))
        write_json(archive/'artifact_manifest.json', dict(status='verified'))
        write_json(archive/'controller/result.json', dict(complete=True))
        (archive/'dual_machine').mkdir()
        (archive/'dual_machine/remote_cleanup_status.txt').write_text('0')
    workspace = tmp_path/'current/controller/codex_workspace'; workspace.mkdir(parents=True)
    prompt = m.inject('BASE', workspace, 'arrange_largest_number_random', e)
    assert all(f'Experience {i}:' in prompt for i in range(7))
    assert 'Experience 8:' not in prompt
    selected = json.loads((workspace/'task_memory_input.json').read_text())['summaries']
    assert len(selected) == 7
    assert [x['finished_at'] for x in selected] == [6, 5, 4, 3, 2, 1, 0]
    assert m.inject('BASE', workspace, 'fold_clothes', e).endswith('No previous summaries are available.\n')
    assert m.inject('BASE', workspace, 'x', {}) == 'BASE'
    with pytest.raises(ValueError): m.enabled(dict(e, ROLLOUT_CONTROLLER_ROUTE='baseline'))


def setup_worker(tmp_path, monkeypatch, events):
    for key, value in env(tmp_path).items(): monkeypatch.setenv(key, value)
    controller = tmp_path/'controller'; workspace = controller/'codex_workspace'; workspace.mkdir(parents=True)
    write_json(controller/'result.json', dict(complete=True, success=False, step_id=1500))
    write_json(controller/'token_usage.json', dict(usage=dict(total=dict(totalTokens=100))))
    transport = Transport(events)
    worker = SimpleNamespace(workspace=workspace, transport=transport, timeout=900, thread_id='thread')
    worker._turn = lambda text: transport.request('turn/start', dict(input=text), worker.timeout)['turn']['id']
    correction = row()
    correction['response']['mode'] = 'eef'
    correction['executed_steps'] = 60
    return worker, SimpleNamespace(history=[correction])


@pytest.mark.parametrize('kind', ['good', 'success', 'missing_label', 'oversize', 'tool', 'transport'])
def test_summary_bounded_best_effort_no_physical_calls(tmp_path, monkeypatch, kind):
    body = 'Correction outcome: failed_or_unverified\nOverview. Main difficulty: grasp. No method verified.'
    if kind == 'success': body = 'Correction outcome: successful_correction\nLater observations confirmed recovery.'
    if kind == 'missing_label': body = 'No supported recovery.'
    if kind == 'oversize': body = 'word ' * 1000
    events = [event('thread/tokenUsage/updated', tokenUsage=dict(total=dict(totalTokens=125))),
              event('item/completed', item=dict(type='agentMessage', id='a', text=body)),
              completed('t0', 'completed', None)]
    if kind == 'tool': events = [dict(event('item/tool/call', tool='robodojo_execute'), id=7)]
    if kind == 'transport': events = [TimeoutError('network')]
    w, r = setup_worker(tmp_path, monkeypatch, events)
    m.summarize(w, r, 'arrange_largest_number')
    record = json.loads((w.workspace.parent/'memory/episode_summary.json').read_text())
    assert record['status'] == {'good':'completed', 'success':'completed', 'missing_label':'invalid_outcome_label', 'oversize':'invalid_length',
                                'tool':'failed', 'transport':'failed'}[kind]
    assert (w.workspace.parent/'memory/episode_summary.md').exists() == (kind in ('good', 'success'))
    assert len(w.transport.requests) == 1
    assert json.loads((w.workspace.parent/'token_usage.json').read_text())['usage']['total']['totalTokens'] == 100
    if kind == 'good': assert record['token_delta']['totalTokens'] == 25
    if kind == 'tool': assert w.transport.replies[0][1]['success'] is False


@pytest.mark.parametrize('steps', [0, 59])
def test_short_correction_skips_paid_summary(tmp_path, monkeypatch, steps):
    w, r = setup_worker(tmp_path, monkeypatch, [])
    r.history[0]['executed_steps'] = steps
    m.summarize(w, r, 'arrange_largest_number')
    assert not w.transport.requests
    record = m.read(w.workspace.parent/'memory/episode_summary.json')
    assert record['status'] == 'skipped_below_correction_threshold'
    assert record['max_correction_streak'] == steps


def test_stats_and_focus_use_same_actual_streak():
    rows = [row() for _ in range(5)]
    for i, (r, mode, steps) in enumerate(zip(rows, ['eef','student','edit','student','eef'], [30,0,30,5,30])):
        r.update(decision=i, executed_steps=steps)
        r['response']['mode'] = mode
    stats = m.correction_stats(rows)
    assert stats['max_correction_streak'] == 60
    assert stats['correction_ratio'] == 90/95
    assert m.focus_packet(rows)['focus_decisions'] == [0,2]
    rows[1]['executed_steps'] = 1
    assert m.correction_stats(rows)['max_correction_streak'] == 30


def test_recent_four_plus_distinct_highest_ratio_four():
    candidates = [dict(path=str(i), finished_at=i, correction_ratio=(10-i)/10) for i in range(10)]
    selected = m.select_summaries(candidates)
    assert [r['path'] for r in selected] == ['9','8','7','6','0','1','2','3']
    assert len({r['path'] for r in selected}) == 8
    for r in candidates: r['correction_ratio'] = .5
    assert [r['path'] for r in m.select_summaries(candidates)] == ['9','8','7','6','5','4','3','2']


def test_incomplete_episode_no_paid_summary(tmp_path, monkeypatch):
    w, r = setup_worker(tmp_path, monkeypatch, [])
    write_json(w.workspace.parent/'result.json', dict(complete=False))
    m.summarize(w, r, 'arrange_largest_number')
    assert not w.transport.requests
    assert json.loads((w.workspace.parent/'memory/episode_summary.json').read_text())['status'] == 'skipped_incomplete'

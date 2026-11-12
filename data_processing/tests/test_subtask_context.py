"""Task context and automatic assistance must not become success evidence."""
import pytest

from data_processing.tests.test_subtasks import episode, candidate
from data_processing.subtasks.prepare import prepare, model_payload
from data_processing.subtasks.contract import collect_errors, validate_and_enrich
from data_processing.subtasks.pipeline import prepare_batch, load_input
from data_processing.subtasks.io import read, write
from hybrid_rollout.robodojo.skill_auto_infer.subtask_instruction import render_instruction


def add_reset(archive):
    write(archive / 'sim/reset.json', dict(episode_id='test-episode',
          instruction='Place the objects in the container with their marked sides upward.',
          metadata=dict(private_score_formula='must not enter the prompt')))


def add_hold(archive):
    history = read(archive / 'controller/history.json')
    history[0].update(force_hold_steps=3, tighten_steps=1, force_hold_events=[dict(
        closing_arms=['left'], steps=[2, 3, 4], actual_count=3,
        end_reason='contact_and_finger_stability_confirmed',
        contact_actors={'left': '/World/private/object'}, samples=[{'force': 30}])])
    write(archive / 'controller/history.json', history)


def test_task_context_is_available_before_action_but_not_outcome_evidence(episode, candidate):
    add_reset(episode)
    prepared = prepare(episode)
    payload = model_payload(prepared)
    assert payload['episode_context']['episode.task_instruction'].startswith('Place the objects')
    assert 'private_score_formula' not in str(payload)
    candidate['segments'][0]['instruction']['source_refs'].append('episode.task_instruction')
    first = validate_and_enrich(prepared, candidate)['segments'][0]
    assert first['instruction']['label_origin'] == 'online_supported'
    candidate['segments'][0]['outcome_evidence_refs'] = ['episode.task_instruction']
    assert any(e['code'] == 'missing_outcome_observation' for e in collect_errors(prepared, candidate))


def test_reset_instruction_is_fingerprinted(episode, tmp_path):
    add_reset(episode)
    batch = tmp_path / 'batch'
    manifest = prepare_batch([episode], batch)
    row = manifest['episodes'][0]
    load_input(batch, row)
    initial = read(episode / 'sim/reset.json')
    initial['instruction'] = 'Changed task'
    write(episode / 'sim/reset.json', initial)
    with pytest.raises(ValueError, match='Source changed'):
        load_input(batch, row)


def test_hold_frames_preserve_timeline_and_control_not_success(episode, candidate):
    add_hold(episode)
    prepared = prepare(episode)
    payload = model_payload(prepared)
    first = payload['decisions'][0]
    assert (first['start_step'], first['end_step'], first['control_type']) == (0, 4, 'pi05')
    assert first['after']['d0000.automatic_grasp_assistance.hold_intervals'] == [
        dict(arms=['left'], executed_action_steps=[1, 2, 3])]
    assert 'contact_and_finger_stability_confirmed' not in str(payload)
    assert '/World/' not in str(payload)
    result = validate_and_enrich(prepared, candidate)
    assert [(s['start_step'], s['end_step']) for s in result['segments']] == [(0, 4), (4, 10)]
    assert result['segments'][0]['control_step_counts'] == {'pi05': 4}
    candidate['segments'][0]['outcome_evidence_refs'] = ['d0000.automatic_grasp_assistance.force_hold_steps']
    assert any(e['code'] == 'missing_outcome_observation' for e in collect_errors(prepared, candidate))


@pytest.mark.parametrize('text,valid', [('x' * 240, True), ('x' * 241, False), ('Move\rturn', False), ('Move\nturn', False)])
def test_custom_limits_match_online(episode, candidate, text, valid):
    segment = candidate['segments'][1]
    segment['retry_of'] = None
    segment['instruction'].update(skill_type='custom', custom_description=text)
    errors = collect_errors(prepare(episode), candidate)
    assert (not any(e['code'] == 'custom_description' for e in errors)) == valid
    online = {k: v for k, v in segment['instruction'].items() if k != 'source_refs'}
    if valid:
        assert render_instruction(online)
    else:
        with pytest.raises(ValueError):
            render_instruction(online)


def test_legacy_archive_has_explicitly_empty_context(episode):
    assert model_payload(prepare(episode))['episode_context'] == {}

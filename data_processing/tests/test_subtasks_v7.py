"""Custom skills, grounded orientation slots and executed-GPT attention reports."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from data_processing.tests.test_subtasks import episode, candidate, segment, patch_for
from data_processing.tests.test_subtasks_v6 import reviews_for, unresolved
from data_processing.subtasks.audit import validate_final
from data_processing.subtasks.client import APIClient
from data_processing.subtasks.contract import canonical, collect_errors, gpt_review_report, validate_and_enrich
from data_processing.subtasks.io import read
from data_processing.subtasks.pipeline import annotate_one, load_config, prepare_batch, protocol, run_batch, validate_run
from data_processing.subtasks.prepare import prepare
from data_processing.subtasks.repair import response_schema


def custom_segment(candidate):
    s = candidate['segments'][1]
    s['retry_of'] = None
    s['instruction'].update(skill_type='custom', destination_ref='support', target_relation='to',
        custom_description='Move digit 5 to support while turning it toward the marked axis')
    return s


def test_custom_success_is_eligible_and_has_canonical_instruction(episode, candidate):
    s = custom_segment(candidate)
    p = prepare(episode)
    p['payload']['refs']['d0002.assessment.current_subgoal']['value'] = (
        'Left hand move digit 5 to support while turning it toward the marked axis')
    p['payload']['refs']['d0003.assessment.execution_evidence']['value'] = (
        'Digit 5 reached support and aligned with the marked axis while retained')
    s['instruction']['target_orientation_ref'] = 'marked_axis'
    s['outcome_rationale'] = 'The retained object reached support and the required orientation.'
    original = deepcopy(candidate)
    out = validate_final(p, candidate, dict(reviews=reviews_for(candidate)))['segments'][1]
    assert out['training_eligibility'] == dict(low_level=True, high_level=True)
    assert not out['needs_review']
    assert out['instruction']['text'] == (
        'Left arm: Move digit 5 to support while turning it toward the marked axis. '
        'Target orientation of digit 5: marked axis.')
    assert candidate == original
    assert out['criterion_id'] == 'custom.v4'


@pytest.mark.parametrize('value', [None, '', '   ', 'Move object\nRotate object'])
def test_custom_requires_nonempty_single_line_instruction(episode, candidate, value):
    custom_segment(candidate)['instruction']['custom_description'] = value
    assert any(e['code'] == 'custom_description' for e in collect_errors(prepare(episode), candidate))


def test_predefined_skill_cannot_smuggle_custom_text(episode, candidate):
    candidate['segments'][1]['instruction']['custom_description'] = 'Move something'
    assert any(e['code'] == 'custom_description' for e in collect_errors(prepare(episode), candidate))


@pytest.mark.parametrize('skill', ['approach', 'align', 'grasp', 'transport', 'place', 'push',
                                 'release', 'rotate', 'handover', 'retreat', 'return_home', 'custom'])
def test_all_skills_preserve_orientation_in_training_text(episode, candidate, skill):
    s = candidate['segments'][1]
    s['retry_of'] = None
    i = s['instruction']
    i.update(skill_type=skill, target_orientation_ref='marked_axis')
    if skill in ('transport', 'place', 'push'):
        i.update(destination_ref='support', target_relation='on' if skill == 'place' else 'to')
    if skill == 'handover':
        i.update(arm='both', giver_arm='left', receiver_arm='right')
    if skill == 'custom':
        i['custom_description'] = 'Move digit 5 along the support edge'
    p = prepare(episode)
    p['payload']['refs']['d0002.assessment.current_subgoal']['value'] = (
        'Perform the operation with digit 5 oriented along the marked axis')
    assert not collect_errors(p, candidate)
    text = validate_and_enrich(p, candidate)['segments'][1]['instruction']['text']
    assert 'marked axis' in text and 'digit 5' in text
    if skill == 'rotate':
        assert text == 'Left arm: Rotate digit 5 to marked axis.'


def test_orientation_requires_object_and_sources(episode, candidate):
    s = custom_segment(candidate)
    s['instruction'].update(target_orientation_ref='marked_axis', object_ref=None, source_refs=[])
    codes = {e['code'] for e in collect_errors(prepare(episode), candidate)}
    assert {'skill_slots', 'missing_instruction_sources'} <= codes


def test_custom_preserves_retrospective_and_real_uncertainty_rules(episode, candidate):
    s = custom_segment(candidate)
    s['instruction']['source_refs'].append('d0003.assessment.execution_evidence')
    p = prepare(episode)
    out = validate_final(p, candidate, dict(reviews=reviews_for(candidate)))['segments'][1]
    assert out['training_eligibility'] == dict(low_level=True, high_level=False)
    report = reviews_for(candidate)
    report[1].update(verdict='needs_review', explanation='The orientation result remains ambiguous.')
    out = validate_final(p, candidate, dict(reviews=report))
    assert not out['segments'][1]['training_eligible']
    assert gpt_review_report(out)['counts']['gpt_success_excluded'] == 1
    unresolved(s)
    out = validate_final(p, candidate, dict(reviews=reviews_for(candidate)))
    assert gpt_review_report(out)['counts']['gpt_unknown_outcome'] == 1
    assert gpt_review_report(out)['counts']['gpt_unknown_skill'] == 0


def test_gpt_attention_distinguishes_skill_result_and_zero_step(episode, candidate):
    # First segment has an endpoint zero-step GPT refusal but only pi05 executed.
    for s in candidate['segments']:
        s['instruction']['skill_type'] = 'unknown'
    unresolved(candidate['segments'][0])
    result = validate_final(prepare(episode), candidate, dict(reviews=reviews_for(candidate)))
    report = gpt_review_report(result)
    assert report['counts'] == dict(gpt_unknown_skill=1, gpt_unknown_outcome=0, gpt_success_excluded=1)
    assert report['flagged_gpt_steps'] == 6 and report['flagged_segments'] == 1
    assert report['segments'][0]['segment_id'] == 's2'
    assert report['segments'][0]['start_step'] == 4
    # A continuous mixed-control segment is still flagged, without double counting steps.
    merged = segment('mixed', 0, 10)
    merged['instruction']['skill_type'] = 'unknown'
    unresolved(merged)
    out = validate_and_enrich(prepare(episode), dict(segments=[merged]))
    report = gpt_review_report(out)
    assert report['counts']['gpt_unknown_skill'] == report['counts']['gpt_unknown_outcome'] == 1
    assert report['flagged_gpt_steps'] == 6


def test_audit_receives_gpt_priority_and_can_resolve_custom(episode, candidate, tmp_path):
    initial = deepcopy(candidate)
    initial['segments'][1]['retry_of'] = None
    initial['segments'][1]['instruction']['skill_type'] = 'unknown'
    custom_segment(candidate)
    calls = []
    class Fake:
        def complete(self, prompt, payload, correction=None):
            calls.append(correction)
            if correction is None:
                return {'text': json.dumps(initial)}
            assert correction['gpt_priority_review']['counts']['gpt_unknown_skill'] == 1
            assert correction['gpt_priority_review']['counts']['gpt_success_excluded'] == 1
            return {'text': json.dumps(dict(**patch_for(candidate, ['s2']), reviews=reviews_for(candidate)))}
    p, out = prepare(episode), tmp_path / 'audit'
    result, state = annotate_one(p, out, load_config(), protocol(), Fake())
    assert len(calls) == 2 and state == 'annotated'
    assert result['segments'][1]['training_eligible']
    assert gpt_review_report(result)['flagged_segments'] == 0
    assert annotate_one(p, out, load_config(), protocol(), Fake())[1] == 'reused'
    assert len(calls) == 2


def test_batch_persists_and_prints_gpt_attention_on_resume(episode, candidate, tmp_path, monkeypatch, capsys):
    root = tmp_path / 'batch'
    prepare_batch([episode], root)
    candidate['segments'][1].update(retry_of=None)
    candidate['segments'][1]['instruction']['skill_type'] = 'unknown'
    cfg = load_config()
    cfg.update(backend='api', api_mode='chat_completions')
    calls = []
    def complete(self, prompt, payload, correction=None):
        calls.append(correction)
        output = candidate if correction is None else dict(patches=[], reviews=reviews_for(candidate))
        return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(output)))])
    monkeypatch.setattr(APIClient, 'complete', complete)
    result = run_batch(root, cfg)
    row = result['results'][0]
    assert row['gpt_unknown_skill'] == row['gpt_success_excluded'] == 1
    assert row['gpt_unknown_outcome'] == row['custom'] == 0
    report = Path(result['run_root']) / 'episodes' / row['key'] / 'gpt_review.json'
    assert read(report)['segments'][0]['flags'] == ['gpt_unknown_skill', 'gpt_success_excluded']
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    alert = next(e for e in events if e.get('event') == 'gpt_annotation_attention')
    assert alert['severity'] == 'warning' and alert['key'] == row['key']
    assert alert['segments'][0]['gpt_steps'] == 6
    assert validate_run(root, result['run_id'])['failures'] == 0
    report.unlink()
    assert run_batch(root, cfg)['results'][0]['state'] == 'reused'
    assert report.exists() and len(calls) == 2


def test_new_schema_and_prompt_are_consistent():
    spec = protocol()
    assert spec['version'] == 'subtasks.v9'
    for correction in (None, dict(mode='patch'), dict(mode='audit')):
        schema = response_schema(correction, dict(total_steps=10))
        encoded = json.dumps(schema)
        assert 'custom_description' in encoded and 'custom_reason' not in encoded
        assert 'uniqueItems' not in encoded
    text = spec['prompt']
    assert 'only rotate may populate' not in text
    assert 'smallest\npossible custom interval' in text
    assert 'gpt_priority_review' in text

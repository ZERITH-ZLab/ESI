"""The model can change judgments, NEVER the observed input, control source or timeline."""
from collections import Counter
from copy import deepcopy
import jsonschema

from data_processing.subtasks.io import require
from data_processing.subtasks.contract import observed_feedback
from . import VERSION


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def enum(*values):
    return dict(type='string', enum=list(values))


TEXT = dict(type='string', minLength=1)
REFS = dict(type='array', items=TEXT, uniqueItems=True)
REASONS = ('instruction_format', 'instruction_verbosity', 'instruction_action_mismatch',
           'uncertain_alignment', 'conflicting_evidence', 'uncertain_attribution', 'semantic_uncertainty')
LABEL = obj(dict(
    decision_id=dict(type='integer', minimum=0),
    attempt_id=dict(type='integer', minimum=0),
    outcome=enum('success', 'failure', 'unknown', 'interrupted'),
    outcome_basis=enum('direct', 'inferred', 'unresolved'),
    outcome_rationale=TEXT,
    unknown_reason=dict(type=['string', 'null']),
    outcome_evidence_refs=REFS,
    instruction_action_alignment=enum('aligned', 'mismatch', 'uncertain', 'not_applicable'),
    alignment_rationale=TEXT, alignment_evidence_refs=REFS,
    review_reasons=dict(type='array', items=enum(*REASONS), uniqueItems=True)))
SCHEMA = obj(dict(labels=dict(type='array', items=LABEL)))
REVIEW = obj(dict(decision_id=dict(type='integer', minimum=0),
                  verdict=enum('accept', 'needs_review'), explanation=TEXT))
AUDIT_SCHEMA = obj(dict(labels=dict(type='array', items=LABEL),
                        reviews=dict(type='array', items=REVIEW)))


def response_schema(correction, payload):
    schema = deepcopy(AUDIT_SCHEMA if correction and correction.get('mode') == 'audit' else SCHEMA)
    ids = ((correction or {}).get('editable_decision_ids') or payload['required_decision_ids'])
    label = schema['properties']['labels']['items']
    if ids:
        label['properties']['decision_id']['enum'] = ids
    if not correction or correction.get('mode') == 'initial':
        schema['properties']['labels']['minItems'] = 1 if payload['required_decision_ids'] else 0
    # Codex strict Structured Outputs rejects uniqueItems. Keep this constraint
    # in the local validator, not in the schema sent to the service.
    def compatible(value):
        if isinstance(value, dict):
            value.pop('uniqueItems', None)
            for child in value.values():
                compatible(child)
        elif isinstance(value, list):
            for child in value:
                compatible(child)
    compatible(schema)
    return schema


class ValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        first = issues[0]
        super().__init__(f'D{first.get("decision_id", "episode")}: {first["message"]} ({len(issues)} issue(s))')


def collect_errors(prepared, candidate):
    issues = []
    def error(code, message, did=None):
        issues.append(dict(code=code, decision_id=did, message=message))
    for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(candidate):
        error('schema', f'{list(e.absolute_path)}: {e.message}')
    if issues:
        return issues
    decisions = {d['decision_id']: d for d in prepared['payload']['decisions']
                 if d['end_step'] > d['start_step']}
    labels = candidate['labels']
    counts = Counter(x['decision_id'] for x in labels)
    for did in sorted(set(decisions) | set(counts)):
        if counts[did] != 1 or did not in decisions:
            error('coverage', 'Label every executed decision exactly once; do not label zero-step events', did)
    if issues:
        return issues
    by_id = {x['decision_id']: x for x in labels}
    refs = prepared['payload']['refs']
    for x in labels:
        did = x['decision_id']
        d = decisions[did]
        aid = x['attempt_id']
        if aid not in decisions or aid > did or by_id[aid]['attempt_id'] != aid:
            error('attempt', 'attempt_id must be the earliest executed decision of this attempt and refer to itself', did)
        for field in ('outcome_evidence_refs', 'alignment_evidence_refs'):
            missing = [k for k in x[field] if k not in refs]
            if missing:
                error('reference', f'{field} contains nonexistent references: {missing}', did)
        outcome_refs = [refs[k] for k in x['outcome_evidence_refs'] if k in refs]
        if x['outcome'] == 'unknown':
            if x['outcome_basis'] != 'unresolved' or not (x['unknown_reason'] or '').strip():
                error('unknown', 'Unknown requires unresolved basis and the specific missing fact', did)
        else:
            if x['outcome_basis'] == 'unresolved' or x['unknown_reason'] is not None:
                error('basis', 'Known outcome requires direct/inferred basis and null unknown_reason', did)
            externally_interrupted = (x['outcome'] == 'interrupted'
                and d['end_step'] == prepared['payload']['total_steps']
                and bool(prepared['metadata'].get('manual_truncation'))
                and any(r['field'] == 'archive.manual_truncation'
                        and r['available_at_step'] == d['end_step'] for r in outcome_refs))
            if not externally_interrupted and not any(observed_feedback(r) and r['available_at_step'] >= d['end_step'] for r in outcome_refs):
                error('outcome_observation', 'Include post-execution feedback supporting THIS interval; earlier evidence may supplement it. Plans/automatic holds alone do not prove outcome.', did)
        state = x['instruction_action_alignment']
        if (d['conditioning']['subtask'] is None) != (state == 'not_applicable'):
            error('alignment', 'Use not_applicable only for an actual null subtask; never invent null instructions', did)
        if state in ('mismatch', 'uncertain') and not x['alignment_evidence_refs']:
            error('alignment', 'Mismatch/uncertain requires grounded references explaining the material conflict', did)
    return issues


def validate_candidate(prepared, candidate):
    issues = collect_errors(prepared, candidate)
    if issues:
        raise ValidationError(issues)
    return candidate


def merge_patch(candidate, patch, editable):
    errors = list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(patch))
    if errors:
        raise ValidationError([dict(code='patch_schema', decision_id=None, message=errors[0].message)])
    ids = [x['decision_id'] for x in patch['labels']]
    if len(set(ids)) != len(ids) or not set(ids).issubset(editable):
        raise ValidationError([dict(code='patch_scope', decision_id=None,
            message='Patch must contain unique authorized decision IDs; boundaries and instructions are immutable')])
    result = {x['decision_id']: deepcopy(x) for x in candidate['labels']}
    result.update({x['decision_id']: deepcopy(x) for x in patch['labels']})
    return dict(labels=[result[k] for k in sorted(result)])


def validate_final(prepared, candidate, report):
    validate_candidate(prepared, candidate)
    schema = obj(dict(reviews=dict(type='array', items=REVIEW)))
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    ids = [x['decision_id'] for x in report.get('reviews', [])] if not errs else []
    expected = {x['decision_id'] for x in candidate['labels']}
    if errs or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValidationError([dict(code='audit_coverage', decision_id=None,
                                    message='Audit must review every executed decision exactly once')])
    reviews = {x['decision_id']: x for x in report['reviews']}
    decisions = {d['decision_id']: d for d in prepared['payload']['decisions']}
    refs = prepared['payload']['refs']
    segments = []
    for label in sorted(candidate['labels'], key=lambda x: x['decision_id']):
        d = decisions[label['decision_id']]
        c = deepcopy(d['conditioning'])
        reasons = set(label['review_reasons'])
        if d['input_format_issues']:
            reasons.add('instruction_format')
        alignment = label['instruction_action_alignment']
        if alignment == 'mismatch':
            reasons.add('instruction_action_mismatch')
        elif alignment == 'uncertain':
            reasons.add('uncertain_alignment')
        if reviews[label['decision_id']]['verdict'] == 'needs_review':
            reasons.add('semantic_uncertainty')
        needs_review = bool(reasons) or label['outcome'] == 'unknown'
        eligible = label['outcome'] == 'success' and not needs_review and d['control_type'] in ('pi05', 'gpt')
        # Compatibility presentation for the existing offline annotation viewer.
        # This is NOT a regenerated training prompt; online_conditioning is authoritative.
        instruction = deepcopy(c['subtask']) if isinstance(c['subtask'], dict) else dict(
            skill_type='unconditioned', arm='unknown')
        instruction.update(text=c['subtask_text'], label_origin='recorded_online')
        segment = dict(**deepcopy(label), segment_id=f'd{label["decision_id"]:04d}',
            start_step=d['start_step'], end_step=d['end_step'],
            online_conditioning=c, instruction=instruction, input_format_issues=d['input_format_issues'],
            control_step_counts={d['control_type']: d['end_step']-d['start_step']},
            control_type=d['control_type'], annotation_status='validated',
            outcome_evidence=[dict(ref_id=k, **refs[k]) for k in label['outcome_evidence_refs']],
            alignment_evidence=[dict(ref_id=k, **refs[k]) for k in label['alignment_evidence_refs']],
            evidence_level=label['outcome_basis'], needs_review=needs_review,
            training_eligible=eligible,
            training_eligibility=dict(low_level=eligible, high_level=eligible and c['subtask'] is not None),
            semantic_review=reviews[label['decision_id']])
        segment['review_reasons'] = sorted(reasons)
        segments.append(segment)
    return dict(version=VERSION, route='online_subtask_outcome', input_hash=prepared['input_hash'],
        metadata=deepcopy(prepared['metadata']), total_steps=prepared['payload']['total_steps'],
        segments=segments, events=deepcopy(prepared['events']),
        semantic_audit=dict(status='completed', reviewed_segments=len(segments)),
        episode_validation='passed',
        training_input_source='online_conditioning.effective_prompt')

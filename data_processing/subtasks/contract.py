"""Strict model schema, canonical wording, temporal validation and enrichment."""
from collections import Counter
from copy import deepcopy
import re

import jsonschema

from . import ANNOTATION_VERSION
from .evidence import endpoint_windows

SKILLS = ("approach", "align", "grasp", "transport", "place", "push", "release", "rotate",
          "handover", "retreat", "return_home", "custom", "unknown")


def obj(properties):
    return dict(type="object", properties=properties, required=list(properties), additionalProperties=False)


def enum(values):
    return dict(type="string", enum=list(values))


STRING = dict(type="string")
NULL_STRING = dict(type=["string", "null"])
ENTITY = dict(type=["string", "null"], pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
REFS = dict(type="array", items=STRING)
INTEGER = dict(type="integer", minimum=0)
INSTRUCTION = obj(dict(
    skill_type=enum(SKILLS), arm=enum(("left", "right", "both", "unknown")),
    object_ref=ENTITY, destination_ref=ENTITY,
    target_relation=enum(("none", "to", "on", "in", "near")),
    target_orientation_ref=ENTITY, custom_description=dict(type=["string", "null"], maxLength=240),
    giver_arm=enum(("left", "right", "unknown")),
    receiver_arm=enum(("left", "right", "unknown")),
    source_refs=REFS))
SEGMENT = obj(dict(
    segment_id=STRING, start_step=INTEGER, end_step=INTEGER,
    instruction=INSTRUCTION,
    parent_goal=NULL_STRING, purpose=enum(("task", "recovery", "unknown")),
    retry_of=NULL_STRING, recovery_of=NULL_STRING,
    boundary_status=enum(("decision_aligned", "approximate", "unresolved")),
    outcome=enum(("success", "failure", "unknown", "interrupted")),
    outcome_basis=enum(("direct", "inferred", "unresolved")),
    outcome_rationale=STRING, unknown_reason=NULL_STRING,
    outcome_evidence_refs=REFS,
    review_reasons=dict(type="array", items=enum((
        "ambiguous_skill", "ambiguous_arm", "ambiguous_object", "ambiguous_boundary",
        "conflicting_evidence", "insufficient_evidence", "attempt_ambiguity", "concurrent_subtasks",
        "other")))))
SCHEMA = obj(dict(segments=dict(type="array", items=SEGMENT)))


def canonical(i):
    """Only code writes training text. Missing slots produce no instruction."""
    skill, arm, target = i["skill_type"], i["arm"], i["object_ref"]
    if skill == "unknown" or arm == "unknown":
        return None
    label = lambda x: x.replace("_", " ") if x else None
    target, destination = label(target), label(i["destination_ref"])
    prefix = {"left": "Left arm", "right": "Right arm", "both": "Both arms"}[arm]
    orientation = label(i["target_orientation_ref"])

    def oriented(text):
        # The field always describes the OBJECT, never an inferred wrist pose.
        if orientation:
            if not target:
                return None
            return text + f" Target orientation of {target}: {orientation}."
        return text

    if skill == "custom":
        description = i["custom_description"]
        if not description or not description.strip():
            return None
        return oriented(f"{prefix}: {description.strip().rstrip('.')}.")
    if skill == "return_home":
        return oriented(f"{prefix}: Return home.")
    if not target:
        return None
    if skill == "handover":
        giver, receiver = i["giver_arm"], i["receiver_arm"]
        if arm != "both" or giver == "unknown" or receiver == "unknown" or giver == receiver:
            return None
        return oriented(f"Transfer {target} from the {giver} gripper to the {receiver} gripper.")
    if skill in ("transport", "place", "push", "reposition"):
        if not destination or i["target_relation"] == "none":
            return None
        return oriented(f"{prefix}: {skill.capitalize()} {target} {i['target_relation']} {destination}.")
    if skill == "rotate":
        return f"{prefix}: Rotate {target} to {orientation}." if orientation else None
    verb = {"approach": "Approach", "align": "Align with", "grasp": "Grasp",
            "retreat": "Retreat from", "release": "Release"}[skill]
    return oriented(f"{prefix}: {verb} {target}.")


class AnnotationValidationError(ValueError):
    """All machine-checkable issues, with stable codes and exact locations."""
    def __init__(self, errors):
        self.errors = errors
        first = errors[0]
        super().__init__(f"{first.get('segment_id') or 'episode'}: {first['message']} "
                         f"({len(errors)} validation issue(s); see validation report)")


def observed_feedback(ref):
    field = ref["field"]
    return (field in ("assessment.execution_evidence", "assessment.execution_status")
            or field.startswith("assessment.task_progress.verified_completed.")
            or (ref["phase"] == "after" and field.endswith((".status", ".error"))))


def collect_errors(prepared, response):
    """No label mutation: report all issues instead of failing on the first segment.

    References and coverage are machine-checkable. Skill completion and attribution
    to THIS attempt are checked by the semantic audit, not an endpoint whitelist.
    """
    errors = []
    for error in jsonschema.Draft202012Validator(SCHEMA).iter_errors(response):
        path = list(error.absolute_path)
        # Entity patterns and custom-text length are local label errors checked
        # below. Other schema violations remain structural.
        if error.validator == "pattern" or (error.validator == "maxLength" and path[-1:] == ["custom_description"]):
            continue
        sid = None
        if len(path) > 1 and path[0] == "segments" and isinstance(response, dict):
            row = response["segments"][path[1]]
            sid = row.get("segment_id") if isinstance(row, dict) else None
        errors.append(dict(code="schema", segment_id=sid, field=".".join(map(str, path)),
                           message=error.message))
    # Semantic checks depend on the schema. All schema errors are still reported.
    if errors:
        return errors
    data = prepared["payload"]
    refs, decisions = data["refs"], data["decisions"]
    boundaries = {0, data["total_steps"]} | {d[k] for d in decisions for k in ("start_step", "end_step")}
    cursor, seen = 0, {}
    for index, s in enumerate(response["segments"]):
        sid, start, end = s["segment_id"], s["start_step"], s["end_step"]

        def check(condition, code, field, message, **context):
            if not condition:
                errors.append(dict(code=code, segment_id=sid, segment_index=index,
                                   field=field, message=message, **context))

        check(sid and sid not in seen, "segment_identity", "segment_id", "Segment IDs must be nonempty and unique")
        check(start == cursor and start < end and start in boundaries and end in boundaries,
              "partition", "start_step/end_step", "Segments must partition executed steps at decision boundaries",
              expected_start=cursor, start_step=start, end_step=end)
        i = s["instruction"]
        for slot in ("object_ref", "destination_ref", "target_orientation_ref"):
            check(i[slot] is None or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", i[slot]),
                  "entity_id", f"instruction.{slot}", "Use a normalized entity ID, not a sentence")
        if i["skill_type"] != "handover":
            check(i["giver_arm"] == i["receiver_arm"] == "unknown", "skill_slots", "instruction",
                  "Handover roles on non-handover skill")
        else:
            check(i["arm"] == "both", "skill_slots", "instruction.arm", "Handover requires both arms")
            check(i["giver_arm"] == "unknown" or i["receiver_arm"] == "unknown" or
                  i["giver_arm"] != i["receiver_arm"], "skill_slots", "instruction", "Handover arms must differ")
        if i["skill_type"] in ("approach", "align", "grasp", "release", "retreat", "return_home"):
            check(i["destination_ref"] is None and i["target_relation"] == "none",
                  "skill_slots", "instruction", "Unexpected destination for skill")
        if i["target_orientation_ref"] is not None:
            check(i["object_ref"] is not None, "skill_slots", "instruction.target_orientation_ref",
                  "Object target orientation requires object_ref; never substitute a wrist pose")
            check(bool(i["source_refs"]), "missing_instruction_sources", "instruction.source_refs",
                  "Target orientation requires explicit instruction sources")
        if i["skill_type"] == "custom":
            description = i["custom_description"]
            check(bool(description and description.strip()) and
                  len(description or "") <= 240 and
                  not any(c in (description or "") for c in ("\n", "\r")),
                  "custom_description", "instruction.custom_description",
                  "Custom skill requires a concise single-line action instruction of at most 240 characters grounded in source_refs")
        else:
            check(i["custom_description"] is None, "custom_description", "instruction.custom_description",
                  "Only custom skills may have custom_description; use null otherwise")
        check(i["skill_type"] == "unknown" or bool(i["source_refs"]), "missing_instruction_sources",
              "instruction.source_refs", "Known skill requires instruction sources")
        groups = {"instruction.source_refs": i["source_refs"],
                  "outcome_evidence_refs": s["outcome_evidence_refs"]}
        for field, keys in groups.items():
            check(len(keys) == len(set(keys)), "duplicate_reference", field, "Duplicate source reference")
            for key in keys:
                check(key in refs, "missing_reference", field, f"nonexistent source reference {key}", ref_id=key)
        outcome_refs = [refs[k] for k in s["outcome_evidence_refs"] if k in refs]
        check(bool(s["outcome_rationale"].strip()), "missing_outcome_rationale", "outcome_rationale",
              "Explain whether the core goal is met, failed, or unresolved using cited records; "
              "auxiliary completion is not required; explain causal attribution for downstream inference")
        if s["outcome"] == "unknown":
            check(s["outcome_basis"] == "unresolved", "outcome_basis", "outcome_basis",
                  "Unknown outcome must have unresolved basis")
            check(bool(s["unknown_reason"] and s["unknown_reason"].strip()), "missing_unknown_reason", "unknown_reason",
                  "Unknown requires the missing decisive fact or unresolved contradiction, not merely an empty quote")
        else:
            check(s["outcome_basis"] in ("direct", "inferred"), "outcome_basis", "outcome_basis",
                  "Definitive outcome must distinguish direct confirmation from inference")
            check(s["unknown_reason"] is None, "unexpected_unknown_reason", "unknown_reason",
                  "Only unknown outcomes have unknown_reason")
            check(bool(outcome_refs), "missing_outcome_evidence", "outcome_evidence_refs",
                  "Definitive outcome requires source references supporting this attempt's result")
            check(any(observed_feedback(r) for r in outcome_refs), "missing_outcome_observation", "outcome_evidence_refs",
                  "Plans/commands alone cannot establish a result; include observed feedback, completion history or execution receipts")
        # Evidence may precede the endpoint or report it later. No timestamp or
        # latest-field rule can decide causality; preserve times for the audit.
        for relation in ("retry_of", "recovery_of"):
            check(s[relation] is None or s[relation] in seen, "invalid_attempt_link", relation,
                  f"{relation} must refer to an earlier segment")
        previous = seen.get(s["retry_of"])
        if previous:
            check(previous["outcome"] in ("failure", "interrupted", "unknown"), "retry_success", "retry_of",
                  "Cannot retry a successful attempt")
            check(all(previous["instruction"][k] == i[k] for k in
                      ("skill_type", "arm", "object_ref", "destination_ref", "target_relation", "target_orientation_ref")),
                  "retry_goal_mismatch", "retry_of", "Retry must keep the same skill, arm, object and target")
            # Custom prose may be paraphrased. The semantic audit verifies that
            # the complete custom goal (including text-only roles) is unchanged.
        seen[sid], cursor = s, end
    if cursor != data["total_steps"]:
        errors.append(dict(code="incomplete_coverage", segment_id=None, field="segments",
                           message="Not all executed steps are annotated", covered_until=cursor,
                           total_steps=data["total_steps"]))
    return errors


def instruction_timing(instruction, start, refs, decisions):
    """Derive provenance, never rewrite the model's skill, object, target or refs."""
    sources = [refs[k] for k in instruction["source_refs"]]
    if not sources:
        return dict(available_at_step=None, available_at_decision=None, label_origin="unknown")
    first = next(d for d in decisions if d["start_step"] == start and d["end_step"] > start)
    known_at_start = all(r["available_at_step"] <= start and
                         (r["decision_id"] < first["decision_id"] or
                          (r["decision_id"] == first["decision_id"] and r["phase"] == "before"))
                         for r in sources)
    return dict(available_at_step=max(r["available_at_step"] for r in sources),
                available_at_decision=max(r["decision_id"] for r in sources),
                label_origin="online_supported" if known_at_start else "retrospective")


def execution_spans(prepared, start, end):
    spans, counts = [], Counter()
    for d in prepared["payload"]["decisions"]:
        a, b = max(start, d["start_step"]), min(end, d["end_step"])
        if b > a:
            spans.append(dict(start_step=a, end_step=b, decision_id=d["decision_id"],
                              control_type=d["control_type"], source=d["source"]))
            counts[d["control_type"]] += b - a
    return spans, dict(counts)


def gpt_review_report(annotation):
    """Attention flags, not validation failures or automatic success relabels.

    Uses executed spans only: mixed segments count, zero-control proposals do not.
    Also accepts saved v6 annotations for read-only diagnostics.
    """
    counts = dict(gpt_unknown_skill=0, gpt_unknown_outcome=0, gpt_success_excluded=0)
    items = []
    for s in annotation["segments"]:
        steps = s["control_step_counts"].get("gpt", 0)
        if not steps:
            continue
        skill = (s.get("instruction") or {}).get("skill_type", "unknown")
        flags = []
        if skill == "unknown":
            flags.append("gpt_unknown_skill")
        if s["outcome"] == "unknown":
            flags.append("gpt_unknown_outcome")
        if s["outcome"] == "success" and not s["training_eligibility"]["low_level"]:
            flags.append("gpt_success_excluded")
        if not flags:
            continue
        for flag in flags:
            counts[flag] += 1
        items.append(dict(segment_id=s["segment_id"], start_step=s["start_step"], end_step=s["end_step"],
                          gpt_steps=steps, skill_type=skill, outcome=s["outcome"], flags=flags,
                          review_reasons=s["review_reasons"], unknown_reason=s.get("unknown_reason"),
                          outcome_rationale=s.get("outcome_rationale"),
                          semantic_review=s.get("semantic_review")))
    return dict(episode_id=annotation.get("metadata", {}).get("episode_id"),
                counts=counts, flagged_segments=len(items),
                flagged_gpt_steps=sum(s["gpt_steps"] for s in items), segments=items)


def classify_exclusions(prepared, response, errors):
    """Quarantine local pi05 label errors; never relax structure or trust dependencies."""
    hard = {"schema", "segment_identity", "partition", "incomplete_coverage", "invalid_attempt_link"}
    if any(e["code"] in hard or not e.get("segment_id") for e in errors):
        return errors, {}
    rows = {s["segment_id"]: s for s in response["segments"]}
    local, blocking = {}, []
    def pure(s):
        _, counts = execution_spans(prepared, s["start_step"], s["end_step"])
        return counts == {"pi05": s["end_step"] - s["start_step"]}
    for e in errors:
        s = rows.get(e["segment_id"])
        if s is not None and pure(s):
            local.setdefault(s["segment_id"], []).append(e)
        else:
            blocking.append(e)
    # Relations cannot rely on a quarantined label's outcome/goal. Cascade only
    # through pure-pi05 labels; GPT/mixed dependents must be repaired explicitly.
    changed = True
    dependencies = set()
    while changed:
        changed = False
        for sid, s in rows.items():
            if sid in local:
                continue
            for field in ("retry_of", "recovery_of"):
                target = s[field]
                if target not in local or (sid, field) in dependencies:
                    continue
                dependencies.add((sid, field))
                e = dict(code="excluded_dependency", segment_id=sid, field=field,
                         message="Attempt link depends on an unvalidated pi05 label; repair or justify removing the link",
                         target_segment_id=target)
                if pure(s):
                    local.setdefault(sid, []).append(e)
                    changed = True
                else:
                    blocking.append(e)
    return blocking, local


def validate_and_enrich(prepared, response, *, allow_pi05_exclusions=False):
    errors = collect_errors(prepared, response)
    blocking, excluded = (classify_exclusions(prepared, response, errors)
                          if allow_pi05_exclusions else (errors, {}))
    if blocking:
        raise AnnotationValidationError(blocking)
    data, segments = prepared["payload"], []
    refs, decisions = data["refs"], data["decisions"]
    windows = endpoint_windows(data)

    def resolve(keys):
        return [dict(ref_id=k, text=refs[k]["value"] if isinstance(refs[k]["value"], str) else None,
                     **refs[k]) for k in keys]

    for raw in response["segments"]:
        s = deepcopy(raw)
        i, start, end = s["instruction"], s["start_step"], s["end_step"]
        control_spans, counts = execution_spans(prepared, start, end)
        if s["segment_id"] in excluded:
            # Retain the original draft for audit, but do not expose invalid
            # claims as a successful outcome or a usable instruction.
            segments.append(dict(segment_id=s["segment_id"], start_step=start, end_step=end,
                                 annotation_status="excluded_pi05", raw_annotation=s,
                                 validation_errors=excluded[s["segment_id"]],
                                 instruction=None, outcome="unknown", outcome_basis="unresolved",
                                 unknown_reason="Annotation excluded after validation failure",
                                 outcome_rationale="Unvalidated pi05 label; original claims retained only in raw_annotation",
                                 outcome_evidence=[],
                                 control_spans=control_spans, control_step_counts=counts,
                                 review_reasons=["annotation_validation_failed"], needs_review=True,
                                 training_eligibility=dict(low_level=False, high_level=False),
                                 training_eligible=False))
            continue
        i.update(instruction_timing(i, start, refs, decisions))
        text, reasons = canonical(i), set(s["review_reasons"])
        if i["arm"] == "unknown":
            reasons.add("ambiguous_arm")
        if i["skill_type"] == "unknown":
            reasons.add("ambiguous_skill")
        if text is None:
            reasons.add("incomplete_instruction")
        if i["label_origin"] == "unknown":
            reasons.add("missing_instruction_sources")
        if s["outcome"] == "unknown":
            reasons.add("unresolved_outcome")
        if s["boundary_status"] != "decision_aligned":
            reasons.add("ambiguous_boundary")
        if counts.get("unknown"):
            reasons.add("unknown_control")
        i["text"], i["sources"] = text, resolve(i["source_refs"])
        outcome = resolve(s["outcome_evidence_refs"])
        low_level = s["outcome"] == "success" and not reasons
        high_level = low_level and i["label_origin"] == "online_supported"
        s.update(annotation_status="validated", validation_errors=[],
                 criterion_id=f"{i['skill_type']}.v4", outcome_evidence=outcome,
                 outcome_evidence_context=deepcopy(windows[str(end)]),
                 evidence_timing=[dict(ref_id=r["ref_id"], relation=(
                     "before_segment" if r["available_at_step"] < start else
                     "during_segment" if r["available_at_step"] < end else
                     "endpoint" if r["available_at_step"] == end else "later_feedback")) for r in outcome],
                 assessed_at_step=max((r["available_at_step"] for r in outcome), default=None),
                 evidence_level=("inferred" if s["outcome_basis"] == "inferred" else "reported") if outcome else "none",
                 control_spans=control_spans, control_step_counts=dict(counts),
                 review_reasons=sorted(reasons), needs_review=bool(reasons),
                 information_flags=["retrospective_instruction"] if i["label_origin"] == "retrospective" else [],
                 training_eligibility=dict(low_level=low_level, high_level=high_level),
                 training_eligible=low_level)  # Compatibility alias; never use for high-level training.
        segments.append(s)
    return dict(version=ANNOTATION_VERSION, input_hash=prepared["input_hash"], metadata=prepared["metadata"],
                episode_validation="passed_with_pi05_exclusions" if excluded else "passed",
                excluded_segment_ids=sorted(excluded), validation_errors=errors,
                total_steps=data["total_steps"], segments=segments, events=prepared["events"])

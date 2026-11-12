"""Actionable repair context and host-applied, range-preserving local patches."""
from copy import deepcopy

import jsonschema

from .contract import SEGMENT, SCHEMA, STRING, obj, enum, AnnotationValidationError
from .evidence import annotation_payload

# Shared by structural repairs and every semantic-audit attempt.
CORE_OUTCOME_POLICY = (
    "Use one outcome for the core goal for both single-arm and dual-arm operations. "
    "Choose key versus auxiliary from original intent, not which action succeeded; "
    "judge only the key goal and do not add auxiliary completion conditions. "
    "For equally important compound skills, accept individual confirmations OR a defensible "
    "inference that the shared purpose was achieved from actual downstream parent/sibling feedback. "
    "Separate component reports are not required. Explain the causal link in outcome_rationale; "
    "a corrective redo, new attempt, unrelated success or shared parent alone cannot prove this attempt succeeded. "
    "Do not force success or unknown; remove review reasons based solely on concurrency or auxiliary uncertainty. "
)

PATCH_SCHEMA = obj(dict(patches=dict(type="array", items=obj(dict(
    # strict Structured Outputs rejects uniqueItems; enforce it in merge_patch.
    remove_segment_ids=dict(type="array", items=STRING, minItems=1),
    replacement_segments=dict(type="array", items=SEGMENT, minItems=1),
    root_cause=enum(("evidence_selection", "semantic_boundary", "outcome_judgment", "attempt_link", "format")),
    explanation=dict(type="string", pattern=r"\S"))))))


AUDIT_SCHEMA = obj(dict(patches=PATCH_SCHEMA["properties"]["patches"],
    reviews=dict(type="array", items=obj(dict(segment_id=STRING,
        verdict=enum(("confirmed", "needs_review")), explanation=dict(type="string", pattern=r"\S"))))))


def response_schema(correction=None, payload=None):
    schema = deepcopy(AUDIT_SCHEMA if correction and correction["mode"] == "audit" else
                      PATCH_SCHEMA if correction and correction["mode"] == "patch" else SCHEMA)
    if "segments" in schema["properties"] and payload and payload.get("total_steps", 0) > 0:
        schema["properties"]["segments"]["minItems"] = 1
    return schema


def patchable(previous):
    if not isinstance(previous, dict) or set(previous) != {"segments"} or not isinstance(previous["segments"], list):
        return False
    rows = previous["segments"]
    ids = [s.get("segment_id") if isinstance(s, dict) else None for s in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        return False
    return (all(type(s.get(k)) is int for s in rows for k in ("start_step", "end_step"))
            and all(s.get(k) is None or isinstance(s.get(k), str)
                    for s in rows for k in ("retry_of", "recovery_of")))


def build_correction(prepared, previous, errors):
    payload = annotation_payload(prepared)
    rows = previous["segments"] if patchable(previous) else []
    by_id = {s["segment_id"]: s for s in rows}
    # Invalid coverage/identity cannot be safely handled by outer-range-preserving patches.
    full = not patchable(previous) or any(e["code"] in (
        "segment_identity", "partition", "incomplete_coverage") or not e.get("segment_id") for e in errors)
    affected = {e["segment_id"] for e in errors if e.get("segment_id") in by_id}
    editable = set(affected)
    for n, s in enumerate(rows):
        if s["segment_id"] in affected:
            editable.update(r["segment_id"] for r in rows[max(0, n-1):n+2])
            for field in ("retry_of", "recovery_of"):
                if s.get(field) in by_id:
                    editable.add(s[field])
    # Include dependents transitively so split/merge/outcome repairs cannot strand links.
    while True:
        expanded = editable | {s["segment_id"] for s in rows
                               if any(s.get(f) in editable for f in ("retry_of", "recovery_of"))}
        if expanded == editable:
            break
        editable = expanded
    groups = []
    for sid in sorted(affected):
        s = by_id[sid]
        n = rows.index(s)
        window = payload["endpoint_feedback_contexts"].get(str(s["end_step"]), {})
        refs = prepared["payload"]["refs"]
        related = [d for d in payload["decisions"]
                   if s["start_step"] <= d["start_step"] <= s["end_step"]]
        before = [d for d in payload["decisions"] if d["start_step"] < s["start_step"]]
        if before:
            related.insert(0, before[-1])
        related.extend(d for d in payload["decisions"]
                       if d["start_step"] > s["end_step"] and d["start_step"] <=
                       (rows[n+1]["end_step"] if n+1 < len(rows) else s["end_step"]))
        groups.append(dict(segment_id=sid, segment=s,
                           issues=[e for e in errors if e.get("segment_id") == sid],
                           neighbor_segments=rows[max(0, n-1):n] + rows[n+1:n+2],
                           boundary_decisions=related, endpoint_context=window,
                           endpoint_evidence=[dict(ref_id=k, role="state_report" if k in window.get(
                               "state_ref_ids", []) else "supporting", **refs[k])
                               for k in window.get("context_ref_ids", [])],
                           linked_segments=[by_id[s[f]] for f in ("retry_of", "recovery_of") if s.get(f) in by_id],
                           dependent_links=[dict(segment_id=r["segment_id"], field=f) for r in rows
                                            for f in ("retry_of", "recovery_of") if r.get(f) == sid]))
    return dict(mode="full" if full else "patch", previous=previous,
                error="; ".join(e["message"] for e in errors), errors=errors, repair_groups=groups,
                editable_segment_ids=sorted(editable),
                repair_policy=CORE_OUTCOME_POLICY + "Check semantic boundaries before selecting evidence. Preserve locked segments; "
                "Use direct or defensible inferred results from relevant records, not an endpoint whitelist. "
                "Distinguish progress from skill completion; explain any remaining unknown. "
                "Check affected neighbors and dependent links. In patch mode return patches only; "
                "otherwise return the full annotation.")


def merge_patch(previous, patch, editable_ids):
    def reject(message):
        raise AnnotationValidationError([dict(code="invalid_patch", segment_id=None,
                                               field="patches", message=message)])
    errors = list(jsonschema.Draft202012Validator(PATCH_SCHEMA).iter_errors(patch))
    if errors:
        reject("Repair patch violates schema: " + errors[0].message)
    rows = previous["segments"]
    index = {s["segment_id"]: n for n, s in enumerate(rows)}
    touched, replacements = set(), []
    for edit in patch["patches"]:
        ids, new = edit["remove_segment_ids"], edit["replacement_segments"]
        if len(set(ids)) != len(ids):
            reject("Patch remove_segment_ids must not contain duplicates")
        if any(sid not in index or sid not in editable_ids or sid in touched for sid in ids):
            reject("Patch targets missing, locked or already replaced segments")
        positions = [index[sid] for sid in ids]
        if positions != list(range(positions[0], positions[-1]+1)):
            reject("Patch targets must be contiguous and in timeline order")
        a, b = positions[0], positions[-1]
        if new[0]["start_step"] != rows[a]["start_step"] or new[-1]["end_step"] != rows[b]["end_step"]:
            reject("Patch must preserve the replaced range's outer boundaries; include affected neighbors")
        touched.update(ids)
        replacements.append((a, b, new))
    merged = deepcopy(rows)
    for a, b, new in sorted(replacements, reverse=True):
        merged[a:b+1] = deepcopy(new)
    # Global validation checks partition, IDs, references and attempt links after merging.
    return dict(segments=merged)

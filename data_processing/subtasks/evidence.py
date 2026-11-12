"""Endpoint observations are navigation context, never an exclusive evidence window."""
from collections import defaultdict


def endpoint_windows(payload):
    """Compatibility function name: v6 returns non-exclusive endpoint contexts.

    Include every before-report at the endpoint until the next actual execution,
    including reasoning interrupted by zero-control rejections. Later and earlier
    evidence remains available in the full ledger for causal attribution.
    """
    refs, by_decision, by_start, executed_by_end = payload["refs"], defaultdict(list), defaultdict(list), {}
    for key, ref in refs.items():
        by_decision[ref["decision_id"]].append(key)
    for d in payload["decisions"]:
        by_start[d["start_step"]].append(d)
        if d["end_step"] > d["start_step"]:
            executed_by_end[d["end_step"]] = d
    result = {}
    for end, last in sorted(executed_by_end.items()):
        reasoning = sorted((d for d in by_start[end] if d["decision_id"] > last["decision_id"]),
                           key=lambda d: d["decision_id"])
        nxt = next((d for d in reasoning if d["end_step"] > end), None)
        if nxt:
            reasoning = [d for d in reasoning if d["decision_id"] <= nxt["decision_id"]]
        state, support = [], []
        for d in reasoning:
            for k in by_decision[d["decision_id"]]:
                r = refs[k]
                if r["phase"] != "before" or r["available_at_step"] != end:
                    continue
                if r["field"] == "assessment.execution_evidence" and isinstance(r["value"], str) and r["value"].strip():
                    state.append(k)
                elif r["field"] == "assessment.execution_status" or r["field"].startswith("assessment.task_progress.verified_completed."):
                    support.append(k)
        for k in by_decision[last["decision_id"]]:
            r = refs[k]
            if r["phase"] == "after" and r["available_at_step"] == end and r["field"].endswith((".status", ".error")):
                support.append(k)
        result[str(end)] = dict(end_step=end, last_execution_decision_id=last["decision_id"],
            reasoning_decision_ids=[d["decision_id"] for d in reasoning],
            next_execution_decision_id=nxt["decision_id"] if nxt else None,
            state_ref_ids=sorted(state), supporting_ref_ids=sorted(support),
            context_ref_ids=sorted(set(state + support)), exclusive=False,
            state_feedback_status="available" if state else "no_endpoint_state_report")
    return result


def annotation_payload(prepared):
    from .prepare import model_payload
    payload = model_payload(prepared)
    payload["endpoint_feedback_contexts"] = endpoint_windows(prepared["payload"])
    return payload

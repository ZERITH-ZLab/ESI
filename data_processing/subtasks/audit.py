"""One consolidated semantic audit; bounded repair, no heuristic success inference."""
from copy import deepcopy
import jsonschema

from .client import extract_text
from .contract import AnnotationValidationError, validate_and_enrich, gpt_review_report
from .io import read, write
from .evidence import annotation_payload
from .repair import AUDIT_SCHEMA, CORE_OUTCOME_POLICY, merge_patch


def review_error(message, sid=None):
    return AnnotationValidationError([dict(code="semantic_review", segment_id=sid,
                                           field="reviews", message=message)])


def apply_reviews(annotation, report):
    reviews = report.get("reviews") if isinstance(report, dict) else None
    schema = AUDIT_SCHEMA["properties"]["reviews"]
    if list(jsonschema.Draft202012Validator(schema).iter_errors(reviews)):
        raise review_error("Each review requires a segment ID, verdict and nonempty explanation")
    ids = [r["segment_id"] for r in reviews]
    expected = {s["segment_id"] for s in annotation["segments"] if s["annotation_status"] == "validated"}
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise review_error("Review every final non-excluded segment exactly once; update IDs after splits/merges")
    result = deepcopy(annotation)
    by_id = {r["segment_id"]: r for r in reviews}
    for s in result["segments"]:
        if s["segment_id"] not in by_id:
            s["semantic_review"] = dict(verdict="excluded", explanation="Unvalidated pi05 label")
            continue
        s["semantic_review"] = by_id[s["segment_id"]]
        if by_id[s["segment_id"]]["verdict"] == "needs_review":
            s["review_reasons"] = sorted(set(s["review_reasons"]) | {"semantic_uncertainty"})
            s["needs_review"] = True
            s["training_eligibility"] = dict(low_level=False, high_level=False)
            s["training_eligible"] = False
    result["semantic_audit"] = dict(status="completed", reviewed_segments=len(reviews),
        needs_review=sum(r["verdict"] == "needs_review" for r in reviews))
    return result


def validate_final(prepared, candidate, report):
    return apply_reviews(validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True), report)


def audit_semantics(prepared, candidate, annotation, destination, cfg, spec, client, repairs_remaining):
    """One audit call plus the UNUSED part of the shared max_repairs budget.

    Failed audit patches never overwrite the structurally accepted base draft.
    They are supplied as rejected_output with actionable errors on the next call.
    """
    editable = [s["segment_id"] for s in annotation["segments"] if s["annotation_status"] == "validated"]
    if not editable:
        report = dict(reviews=[])
        write(destination / "audit_final.json", report)
        return candidate, apply_reviews(annotation, report)
    correction = dict(mode="audit", previous=candidate, editable_segment_ids=editable,
        excluded_segment_ids=annotation["excluded_segment_ids"], errors=[], rejected_output=None,
        gpt_priority_review=gpt_review_report(annotation),
        repair_policy=CORE_OUTCOME_POLICY + "Audit each non-excluded skill attempt using its core-goal completion criterion. "
        "Reconsider unknown and inferred labels, missed direct outcomes, continuous attempts cut before verification, "
        "progress-only claims and attribution of later observations. Return patches plus reviews for ALL final "
        "non-excluded segments. Confirm justified unknown with its missing fact; never force binary guesses. "
        "Prefer predefined skills, but preserve meaningful granularity and use grounded custom skills when needed. "
        "Prioritize gpt_priority_review: inspect taxonomy gaps, control-handoff boundaries and overlooked feedback; "
        "do not force success or fabricate a custom goal. Verify every populated object orientation is an explicit "
        "intended core target/constraint, and every custom description is grounded, concise and expresses its core goal. "
        "Do not flag a skill solely because it is custom or compound. "
        "Patches always apply to previous, not rejected_output. Preserve excluded segments.")
    payload = annotation_payload(prepared)
    for attempt in range(repairs_remaining + 1):
        write(destination / f"audit_request_{attempt:02d}.json", correction)
        response_path = destination / f"audit_response_{attempt:02d}.json"
        if response_path.exists():
            raw = read(response_path)
        else:
            raw = client.complete(spec["prompt"], payload, correction)
            write(response_path, raw)
        output = None
        try:
            output = extract_text(raw, "codex" if cfg["backend"] == "codex" else cfg["api_mode"])
            write(destination / f"audit_draft_{attempt:02d}.json", output)
            schema_errors = list(jsonschema.Draft202012Validator(AUDIT_SCHEMA).iter_errors(output))
            if schema_errors:
                raise review_error("Invalid audit schema: " + schema_errors[0].message)
            merged = merge_patch(candidate, {"patches": output["patches"]}, editable)
            enriched = validate_and_enrich(prepared, merged, allow_pi05_exclusions=True)
            new_exclusions = set(enriched["excluded_segment_ids"]) - set(annotation["excluded_segment_ids"])
            if new_exclusions and attempt < repairs_remaining:
                raise AnnotationValidationError([e for s in enriched["segments"] if s["segment_id"] in new_exclusions
                                                 for e in s["validation_errors"]])
            report = dict(reviews=output["reviews"])
            enriched = apply_reviews(enriched, report)
            write(destination / f"audit_validation_{attempt:02d}.json", dict(valid=True, errors=[]))
            write(destination / "audit_final.json", report)
            return merged, enriched
        except (ValueError, KeyError, TypeError) as exc:
            errors = exc.errors if isinstance(exc, AnnotationValidationError) else [dict(
                code="semantic_review", segment_id=None, field="response", message=str(exc))]
            write(destination / f"audit_validation_{attempt:02d}.json", dict(valid=False, errors=errors))
            if attempt == repairs_remaining:
                raise ValueError(f"Semantic audit failed after {attempt + 1} calls: {exc}") from None
            correction.update(errors=errors, rejected_output=output if output is not None else raw)
    raise AssertionError("unreachable")

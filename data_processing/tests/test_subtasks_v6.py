"""V6 outcome attribution and mandatory audit integration (no live model calls)."""
from copy import deepcopy
import json

import pytest
import jsonschema

from data_processing.tests.test_subtasks import episode, candidate, segment, patch_for
from data_processing.subtasks.audit import apply_reviews, validate_final
from data_processing.subtasks.client import APIClient, CodexClient
from data_processing.subtasks.contract import collect_errors, validate_and_enrich
from data_processing.subtasks.evidence import annotation_payload
from data_processing.subtasks.io import read
from data_processing.subtasks.pipeline import annotate_one, load_config, protocol, prepare_batch, run_batch, validate_run
from data_processing.subtasks.prepare import prepare
from data_processing.subtasks.repair import response_schema, AUDIT_SCHEMA


def reviews_for(candidate, verdict="confirmed"):
    return [dict(segment_id=s["segment_id"], verdict=verdict,
                 explanation="The observed object state supports this skill criterion, not merely an intended command.")
            for s in candidate["segments"]]


def unresolved(s):
    s.update(outcome="unknown", outcome_basis="unresolved",
             outcome_rationale="Relevant records do not resolve retention during this attempt.",
             unknown_reason="Whether the object stayed held through the lift is not observed.")


def test_unknown_keeps_conflicting_evidence_and_specific_reason(episode, candidate):
    s = candidate["segments"][0]
    unresolved(s)
    s["outcome_evidence_refs"] = ["d0000.assessment.execution_evidence", "d0002.assessment.execution_evidence"]
    result = validate_and_enrich(prepare(episode), candidate)["segments"][0]
    assert len(result["outcome_evidence"]) == 2
    assert "unresolved_outcome" in result["review_reasons"]
    assert "missing_feedback" not in result["review_reasons"]
    s["unknown_reason"] = " "
    assert any(e["code"] == "missing_unknown_reason" for e in collect_errors(prepare(episode), candidate))


@pytest.mark.parametrize("changes", [dict(outcome_basis="unresolved"), dict(unknown_reason="unexplained"),
                                     dict(outcome_rationale=" ")])
def test_basis_reason_consistency_remains_hard_check(episode, candidate, changes):
    candidate["segments"][1].update(changes)
    assert collect_errors(prepare(episode), candidate)


def test_inferred_success_keeps_training_and_instruction_timing(episode, candidate):
    candidate["segments"][1].update(outcome_basis="inferred",
        outcome_rationale="Closed retention and later object following jointly confirm acquisition.")
    candidate["segments"][1]["instruction"]["source_refs"].append("d0003.assessment.execution_evidence")
    result = validate_final(prepare(episode), candidate, dict(reviews=reviews_for(candidate)))
    s = result["segments"][1]
    assert s["evidence_level"] == "inferred"
    assert s["training_eligibility"] == dict(low_level=True, high_level=False)
    assert s["semantic_review"]["verdict"] == "confirmed"


def test_mandatory_audit_corrects_missed_result_and_resumes(episode, candidate, tmp_path):
    initial = deepcopy(candidate)
    unresolved(initial["segments"][0])
    initial["segments"][0]["outcome_evidence_refs"] = []
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            if correction is None:
                return {"text": json.dumps(initial)}
            assert correction["mode"] == "audit"
            assert correction["editable_segment_ids"] == ["s1", "s2"]
            assert "endpoint_feedback_contexts" in payload
            return {"text": json.dumps(dict(**patch_for(candidate, ["s1"]), reviews=reviews_for(candidate)))}
    client, out = Fake(), tmp_path / "audit"
    result, state = annotate_one(prepare(episode), out, load_config(), protocol(), client)
    assert state == "annotated" and client.calls == 2
    assert result["segments"][0]["outcome"] == "failure"
    assert result["semantic_audit"]["reviewed_segments"] == 2
    assert read(out / "pre_audit_candidate.json") == initial
    assert read(out / "candidate.json") == candidate
    assert read(out / "audit_validation_00.json")["valid"]
    assert annotate_one(prepare(episode), out, load_config(), protocol(), client)[1] == "reused"
    assert client.calls == 2
    (out / "annotation.json").unlink()
    assert annotate_one(prepare(episode), out, load_config(), protocol(), client)[0] == result
    assert client.calls == 2


def test_audit_uncertainty_blocks_training_without_faking_unknown(episode, candidate):
    reviews = reviews_for(candidate)
    reviews[1].update(verdict="needs_review", explanation="The reported lift may concern a later attempt; attribution remains unresolved.")
    s = validate_final(prepare(episode), candidate, dict(reviews=reviews))["segments"][1]
    assert not s["training_eligible"] and not s["training_eligibility"]["high_level"]
    assert "semantic_uncertainty" in s["review_reasons"]


def test_audit_cannot_omit_or_duplicate_reviews(episode, candidate):
    annotation = validate_and_enrich(prepare(episode), candidate)
    for reviews in [[], reviews_for(candidate)[:1], reviews_for(candidate)*2]:
        with pytest.raises(ValueError, match="exactly once"):
            apply_reviews(annotation, dict(reviews=reviews))


def test_audit_uses_remaining_shared_repair_budget(episode, candidate, tmp_path):
    class Fake:
        calls = 0
        repairs = 0
        audits = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            if correction is None:
                bad = deepcopy(candidate)
                bad["segments"][1]["outcome_evidence_refs"] = ["missing"]
                return {"text": json.dumps(bad)}
            if correction["mode"] == "patch":
                self.repairs += 1
                return {"text": json.dumps(patch_for(candidate, ["s2"]))}
            self.audits += 1
            return {"text": json.dumps(dict(patches=[], reviews=reviews_for(candidate) if self.audits == 3 else []))}
    client = Fake()
    result, _ = annotate_one(prepare(episode), tmp_path / "shared-budget", load_config(), protocol(), client)
    assert client.calls == 5 and client.repairs == 1 and client.audits == 3
    assert result["semantic_audit"]["status"] == "completed"


def test_failed_audit_keeps_episode_out_of_index(episode, candidate, tmp_path, monkeypatch):
    root = tmp_path / "bad-audit-batch"
    prepare_batch([episode], root)
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions", max_repairs=0)
    def complete(self, prompt, payload, correction=None):
        output = candidate if correction is None else dict(patches=[], reviews=[])
        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(output)))])
    monkeypatch.setattr(APIClient, "complete", complete)
    result = run_batch(root, cfg)
    from pathlib import Path
    assert result["failures"] == 1
    r = Path(result["run_root"])
    assert read(r / "segments.json") == []
    assert not list(r.glob('episodes/*/annotation.json'))
    assert list(r.glob('episodes/*/audit_validation_00.json'))


def test_excluded_pi05_not_reviewed_but_retained_segments_are(episode, candidate, tmp_path):
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    candidate["segments"][1].update(retry_of=None, recovery_of=None)
    class Fake:
        def complete(self, prompt, payload, correction=None):
            if correction is None:
                return {"text": json.dumps(candidate)}
            assert correction["mode"] == "audit" and correction["editable_segment_ids"] == ["s2"]
            return {"text": json.dumps(dict(patches=[], reviews=reviews_for(candidate)[1:]))}
    cfg = load_config()
    cfg["max_repairs"] = 0
    result, state = annotate_one(prepare(episode), tmp_path / "excluded", cfg, protocol(), Fake())
    assert state == "annotated_with_exclusions"
    assert result["segments"][0]["semantic_review"]["verdict"] == "excluded"
    assert result["segments"][1]["training_eligible"]


def test_zero_episode_has_no_unnecessary_audit_call():
    # Covered by zero-step preparation regression; audit itself must not call a model.
    from data_processing.subtasks.audit import audit_semantics
    import tempfile
    from pathlib import Path
    class Never:
        def complete(self, *args):
            raise AssertionError("Zero-segment episode has nothing to audit")
    with tempfile.TemporaryDirectory() as tmp:
        _, result = audit_semantics({}, {"segments": []}, {"segments": [], "excluded_segment_ids": []},
            Path(tmp), {}, {}, Never(), 3)
    assert result["semantic_audit"]["reviewed_segments"] == 0


def test_audit_schema_is_strict_compatible_and_reviews_required():
    schema = response_schema({"mode": "audit"}, {"total_steps": 10})
    assert schema == AUDIT_SCHEMA
    assert "uniqueItems" not in json.dumps(schema)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors({"patches": []}))


def test_prompt_distinguishes_progress_from_functional_and_object_state_completion():
    text = protocol()["prompt"]
    assert "Progress words" in text and "neither prove nor disprove" in text
    assert "Closure, contact and beginning a" in text
    assert "sufficiently near the target" in text
    assert "NOT exclusive whitelists" in text
    assert "unknown_reason" in text and "no unknown-rate target" in text
    assert "without extra inference" not in text


def test_successful_batch_indexes_audited_results_and_offline_validation(episode, candidate, tmp_path, monkeypatch):
    from pathlib import Path
    root = tmp_path / "audited-batch"
    prepare_batch([episode], root)
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions")
    candidate["segments"][1]["outcome_basis"] = "inferred"
    calls = []
    def complete(self, prompt, payload, correction=None):
        calls.append(correction["mode"] if correction else "initial")
        output = candidate if correction is None else dict(patches=[], reviews=reviews_for(candidate))
        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(output)))])
    monkeypatch.setattr(APIClient, "complete", complete)
    result = run_batch(root, cfg)
    assert result["failures"] == 0 and calls == ["initial", "audit"]
    status = result["results"][0]
    assert status["inferred"] == 1 and status["semantic_reviewed"] == 2
    assert all("semantic_review" in s for s in read(Path(result["run_root"]) / "segments.json"))
    assert validate_run(root, result["run_id"])["failures"] == 0
    assert run_batch(root, cfg)["results"][0]["state"] == "reused"
    assert len(calls) == 2


def test_audit_changed_segment_ids_require_final_reviews(episode, candidate, tmp_path):
    updated = deepcopy(candidate)
    updated["segments"][0]["segment_id"] = "attempt_a"
    updated["segments"][1].update(retry_of="attempt_a", recovery_of="attempt_a")
    class Fake:
        def complete(self, prompt, payload, correction=None):
            if correction is None:
                return {"text": json.dumps(candidate)}
            patch = dict(patches=[dict(remove_segment_ids=["s1", "s2"],
                replacement_segments=updated["segments"], root_cause="attempt_link", explanation="Use consistent attempt identities.")],
                reviews=reviews_for(updated))
            return {"text": json.dumps(patch)}
    result, _ = annotate_one(prepare(episode), tmp_path / "audit-ids", load_config(), protocol(), Fake())
    assert result["segments"][0]["segment_id"] == "attempt_a"
    assert result["segments"][1]["retry_of"] == "attempt_a"
    assert result["semantic_audit"]["reviewed_segments"] == 2


def test_api_audit_uses_audit_schema(monkeypatch):
    import httpx
    monkeypatch.setenv("SUBTASK_API_KEY", "test-key")
    def handler(request):
        body = json.loads(request.content)
        assert body["text"]["format"]["schema"] == AUDIT_SCHEMA
        return httpx.Response(200, json={})
    APIClient(load_config(), httpx.MockTransport(handler)).complete("prompt", {"total_steps": 10}, {"mode": "audit"})

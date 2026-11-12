from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from data_processing.subtasks.client import APIClient, CodexClient, extract_text
from data_processing.subtasks.contract import AnnotationValidationError, collect_errors, validate_and_enrich
from data_processing.subtasks.io import read, write
from data_processing.subtasks.pipeline import annotate_one, load_config, load_input, prepare_batch, protocol, run_batch
from data_processing.subtasks.prepare import prepare, model_payload
from data_processing.subtasks.evidence import annotation_payload, endpoint_windows
from data_processing.subtasks.repair import build_correction, merge_patch, PATCH_SCHEMA


@pytest.fixture
def episode(tmp_path):
    archive = tmp_path / "experiment/task/random/eval_seed_0/layout_0/replica_1/repeat_001/attempt_000"
    write(archive.parent / "sample.json", dict(repeat_id=1, attempts=[dict(archive=str(archive), finished_at=123)]))
    write(archive / "component_reset.json", dict(all_owned_processes_exited=True, ports_released=True))
    write(archive / "sim/summary.json", dict(episode_id="test-episode", step_id=10, success=False, reason="terminal"))
    history = [
        dict(decision=0, start_tick=0, end_tick=4, executed_steps=4, source="student",
             response=dict(mode="student"),
             assessment=dict(current_subgoal="Left arm grasp digit 5", execution_evidence="Initial state")),
        dict(decision=1, start_tick=4, end_tick=4, executed_steps=0, source="gpt_eef",
             response=dict(mode="eef"), motion_result=dict(status="failed", error="IK failed"),
             assessment=dict(current_subgoal="Left arm grasp digit 5", execution_evidence="5 slipped during lift")),
        dict(decision=2, start_tick=4, end_tick=10, executed_steps=6, source="gpt_eef",
             response=dict(mode="eef"), motion_result=dict(status="failed", error="stopped after execution"),
             assessment=dict(current_subgoal="Left arm grasp digit 5", execution_status="failed",
                             execution_evidence="5 remains on the table after slipping from the gripper")),
        dict(decision=3, start_tick=10, end_tick=10, executed_steps=0, source="student",
             response=dict(mode="student"),
             assessment=dict(current_subgoal="Transport digit 5", execution_evidence="5 moved with the left gripper during lift")),
    ]
    write(archive / "controller/history.json", history)
    return archive


def segment(sid="s1", start=0, end=4):
    return dict(segment_id=sid, start_step=start, end_step=end,
                instruction=dict(skill_type="grasp", arm="left", object_ref="digit_5", destination_ref=None,
                                 target_relation="none", target_orientation_ref=None, custom_description=None,
                                 giver_arm="unknown", receiver_arm="unknown",
                                 source_refs=["d0000.assessment.current_subgoal"]),
                parent_goal=None, purpose="task", retry_of=None, recovery_of=None,
                boundary_status="decision_aligned", outcome="failure",
                outcome_basis="direct", outcome_rationale="The recorded object state establishes this attempt result.",
                unknown_reason=None, outcome_evidence_refs=["d0002.assessment.execution_evidence"], review_reasons=[])


@pytest.fixture
def candidate():
    first, second = segment(), segment("s2", 4, 10)
    second["instruction"]["source_refs"] = ["d0002.assessment.current_subgoal"]
    second.update(retry_of="s1", purpose="recovery", recovery_of="s1", outcome="success",
                  outcome_evidence_refs=["d0003.assessment.execution_evidence"])
    return dict(segments=[first, second])


def test_actual_execution_and_evidence(episode, candidate):
    prepared = prepare(episode)
    result = validate_and_enrich(prepared, candidate)
    first, second = result["segments"]
    assert first["control_step_counts"] == {"pi05": 4}
    assert second["control_step_counts"] == {"gpt": 6}
    assert second["instruction"]["text"] == "Left arm: Grasp digit 5."
    assert second["outcome_evidence"][0]["value"] == "5 moved with the left gripper during lift"
    assert second["evidence_level"] == "reported"
    assert second["training_eligible"] is True
    assert result["metadata"]["episode_success"] is False  # Do not discard successful subtasks.
    assert [e["zero_step"] for e in result["events"]] == [True, False, True]
    assert sum(s["end_step"] - s["start_step"] for s in result["segments"]) == 10


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s[0]["instruction"].update(source_refs=[]), "requires instruction"),
    (lambda s: s[0].update(outcome_rationale=""), "Explain whether the core goal"),
    (lambda s: s[0].update(outcome_evidence_refs=["d0001.assessment.current_subgoal"]), "Plans/commands"),
    (lambda s: s[0].update(outcome_evidence_refs=["nonexistent"]), "nonexistent"),
    (lambda s: s[1].update(start_step=5), "partition"),
    (lambda s: s[1].update(end_step=4), "partition"),
    (lambda s: s.pop(), "Not all"),
    (lambda s: s[0].update(outcome="success", outcome_evidence_refs=["d0002.assessment.execution_status"]), "Cannot retry"),
    (lambda s: s[0].update(retry_of="s2"), "earlier"),
])
def test_contract_rejects_invalid_annotations(episode, candidate, mutation, match):
    mutation(candidate["segments"])
    with pytest.raises(ValueError, match=match):
        validate_and_enrich(prepare(episode), candidate)


def test_unknown_and_retrospective_not_training_eligible(episode, candidate):
    candidate["segments"][1]["instruction"].update(arm="unknown", source_refs=["d0003.assessment.current_subgoal"])
    candidate["segments"][1]["retry_of"] = None
    result = validate_and_enrich(prepare(episode), candidate)["segments"][1]
    assert result["needs_review"] and not result["training_eligible"]
    assert result["instruction"]["text"] is None


def test_same_tick_after_reference_is_retrospective(episode, candidate):
    # An AFTER reference from the executing decision cannot become its own intention.
    prepared = prepare(episode)
    prepared["payload"]["refs"]["bad"] = dict(decision_id=0, available_at_step=0, phase="after", field="motion_result.arm", value="left")
    candidate["segments"][0]["instruction"]["source_refs"] = ["bad"]
    result = validate_and_enrich(prepared, candidate)["segments"][0]
    assert result["instruction"]["label_origin"] == "retrospective"
    assert not result["training_eligibility"]["high_level"]


def test_prepare_manifest_and_source_change(episode, tmp_path):
    output = tmp_path / "batch"
    manifest = prepare_batch([episode.parents[6]], output)
    assert len(manifest["episodes"]) == 1 and not manifest["failures"]
    assert prepare_batch([episode.parents[6]], output) == manifest
    row = manifest["episodes"][0]
    load_input(output, row)
    with (episode / "controller/history.json").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="Source changed"):
        load_input(output, row)


def test_bad_timeline_and_active_archive_rejected(episode):
    history = read(episode / "controller/history.json")
    history[2]["executed_steps"] = 7
    write(episode / "controller/history.json", history)
    with pytest.raises(ValueError, match="timeline"):
        prepare(episode)
    sample = read(episode.parent / "sample.json")
    sample["attempts"][0]["finished_at"] = None
    write(episode.parent / "sample.json", sample)
    with pytest.raises(ValueError, match="finalized"):
        prepare(episode)


def test_repair_raw_resume_and_tamper(episode, candidate, tmp_path):
    class Fake:
        calls = 0

        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            value = deepcopy(candidate)
            if self.calls == 1:
                value["segments"][0]["outcome_evidence_refs"] = ["invented"]
            else:
                assert correction["error"]
                value = patch_for(candidate, ["s1"])
            return {"text": json.dumps(value)}

    client, cfg = Fake(), load_config()
    prepared, destination = prepare(episode), tmp_path / "annotation"
    result, state = annotate_one(prepared, destination, cfg, protocol(), client)
    assert state == "annotated" and client.calls == 2
    assert annotate_one(prepared, destination, cfg, protocol(), client)[1] == "reused"
    assert client.calls == 2
    (destination / "annotation.json").unlink()
    assert annotate_one(prepared, destination, cfg, protocol(), client)[0] == result
    assert client.calls == 2  # Resume from saved responses even after interrupted commit.
    result["segments"][0]["outcome"] = "success"
    write(destination / "annotation.json", result)
    with pytest.raises(ValueError, match="corrupted"):
        annotate_one(prepared, destination, cfg, protocol(), client)


def test_http_structured_output_and_auth_error(monkeypatch, candidate):
    cfg = load_config()
    monkeypatch.setenv("SUBTASK_API_KEY", "test-secret")
    def handler(request):
        body = json.loads(request.content)
        assert body["text"]["format"]["strict"] is True and body["store"] is False
        return httpx.Response(200, json=dict(status="completed", output=[dict(type="message", content=[dict(type="output_text", text=json.dumps(candidate))])]))
    raw = APIClient(cfg, httpx.MockTransport(handler)).complete("prompt", {})
    assert extract_text(raw, "responses") == candidate
    client = APIClient(cfg, httpx.MockTransport(lambda r: httpx.Response(401, text="test-secret")))
    with pytest.raises(ValueError, match="HTTP 401") as exc:
        client.complete("prompt", {})
    assert "test-secret" not in str(exc.value)


def test_codex_command_uses_profile_and_keeps_parent_env(monkeypatch, tmp_path, candidate):
    from hybrid_rollout.robodojo.codex_backend import profiles
    monkeypatch.setenv("CODEX_HOME", "/developer-home")
    monkeypatch.setenv("OPENAI_API_KEY", "never-forward")
    monkeypatch.setattr(profiles, "account_home", lambda *args: tmp_path / "managed-home")
    captured = {}
    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
            self.output = Path(command[command.index("--output-last-message") + 1])

        def communicate(self, value, timeout):
            assert "EPISODE LEDGER" in value
            write(self.output, candidate)
    monkeypatch.setattr("data_processing.subtasks.client.subprocess.Popen", FakeProcess)
    raw = CodexClient(load_config(), "codex_a_4", 99, tmp_path / "runtime").complete("prompt", {})
    assert extract_text(raw, "codex") == candidate
    assert captured["env"]["CODEX_HOME"] == str(tmp_path / "managed-home")
    assert "OPENAI_API_KEY" not in captured["env"]
    assert captured["pass_fds"] == (99,)
    assert "--ignore-user-config" in captured["command"]
    import os
    assert os.environ["CODEX_HOME"] == "/developer-home"


def test_batch_resume_without_network(episode, candidate, tmp_path, monkeypatch):
    root = tmp_path / "batch"
    prepare_batch([episode], root)
    cfg = load_config()
    cfg["backend"] = "api"
    cfg["api_mode"] = "chat_completions"
    calls = []
    def complete(*args):
        calls.append(True)
        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(candidate)))])
    monkeypatch.setattr(APIClient, "complete", complete)
    result = run_batch(root, cfg)
    assert result["failures"] == 0
    assert len(read(Path(result["run_root"]) / "segments.json")) == 2
    assert run_batch(root, cfg)["results"][0]["state"] == "reused"
    assert len(calls) == 1


def test_motion_goals_and_inactive_defaults(episode):
    history = read(episode / "controller/history.json")
    history[0]["response"]["target"] = {"left": {"position": [999, 999, 999]}}
    history[2]["response"].update(
        target={"left": {"position": [-999, -999, -999]}},
        motion=dict(tool="robodojo_move", arm="left", waypoints=[dict(position=[0, 0, 1])],
                    tracking=[{"huge": "unused"}]))
    write(episode / "controller/history.json", history)
    prepared = prepare(episode)
    compact = model_payload(prepared)
    refs = prepared["payload"]["refs"]
    assert refs["d0002.response.motion.arm"]["value"] == "left"
    assert "d0000.response.target.left.position" not in refs
    assert "d0002.response.target.left.position" not in refs
    assert not any("tracking" in key for key in refs)
    before = compact["decisions"][2]["before"]
    assert before["d0002.response.motion.waypoints"] == [{"position": [0, 0, 1]}]
    assert sum(len(d["before"]) + len(d["after"]) for d in compact["decisions"]) == len(refs)


def test_all_zero_step_episode(episode):
    history = read(episode / "controller/history.json")[:1]
    history[0].update(end_tick=0, executed_steps=0)
    write(episode / "controller/history.json", history)
    summary = read(episode / "sim/summary.json")
    summary["step_id"] = 0
    write(episode / "sim/summary.json", summary)
    output = validate_and_enrich(prepare(episode), {"segments": []})
    assert output["total_steps"] == 0 and len(output["events"]) == 1


def test_overlarge_input_does_not_call_model(episode, tmp_path):
    class Never:
        def complete(self, *args):
            raise AssertionError("should not call")
    cfg = load_config()
    cfg["max_input_chars"] = 10
    with pytest.raises(ValueError, match="exceeds"):
        annotate_one(prepare(episode), tmp_path / "out", cfg, protocol(), Never())


def test_batch_keeps_other_episodes_after_failure(episode, candidate, tmp_path, monkeypatch):
    import shutil
    other = episode.parent / "attempt_001"
    shutil.copytree(episode, other)
    sample = read(episode.parent / "sample.json")
    sample["attempts"].append(dict(archive=str(other), finished_at=124))
    write(episode.parent / "sample.json", sample)
    summary = read(other / "sim/summary.json")
    summary["episode_id"] = "second"
    write(other / "sim/summary.json", summary)
    root = tmp_path / "batch-fail"
    prepare_batch([episode, other], root)
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions")
    calls = []
    def complete(*args):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("network offline")
        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(candidate)))])
    monkeypatch.setattr(APIClient, "complete", complete)
    result = run_batch(root, cfg)
    assert result["failures"] == 1
    assert [r["state"] for r in result["results"]] == ["failed", "annotated"]
    assert len(read(Path(result["run_root"]) / "segments.json")) == 2


def test_removed_process_evidence_field_is_rejected(episode, candidate):
    candidate["segments"][0]["process_evidence_refs"] = ["d0000.assessment.execution_evidence"]
    errors = collect_errors(prepare(episode), candidate)
    assert any(e["code"] == "schema" and "process_evidence_refs" in e["message"] for e in errors)


def test_unknown_without_direct_final_evidence_is_valid(episode, candidate):
    first = candidate["segments"][0]
    first.update(outcome="unknown", outcome_basis="unresolved", unknown_reason="Retention cannot be resolved from observations.", outcome_evidence_refs=[])
    result = validate_and_enrich(prepare(episode), candidate)
    assert result["episode_validation"] == "passed"
    s = result["segments"][0]
    assert s["outcome"] == "unknown" and s["outcome_evidence"] == []
    assert s["assessed_at_step"] is None and s["evidence_level"] == "none"
    assert "process_evidence" not in s and "process_evidence_refs" not in s
    assert not s["training_eligibility"]["low_level"]


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupted"])
def test_definitive_labels_require_final_evidence(episode, candidate, outcome):
    first = candidate["segments"][0]
    first.update(outcome=outcome, outcome_evidence_refs=[])
    errors = collect_errors(prepare(episode), candidate)
    assert any(e["code"] == "missing_outcome_evidence" for e in errors)


def test_within_attempt_evidence_can_support_inference(episode, candidate):
    candidate["segments"][0]["outcome_evidence_refs"].append("d0000.assessment.execution_evidence")
    candidate["segments"][0]["outcome_basis"] = "inferred"
    result = validate_and_enrich(prepare(episode), candidate)
    assert result["segments"][0]["evidence_timing"][1]["relation"] == "during_segment"


def test_retrospective_success_allowed_only_for_low_level(episode, candidate):
    prepared = prepare(episode)
    second = candidate["segments"][1]
    second["instruction"]["source_refs"].append("d0003.assessment.execution_evidence")
    original = deepcopy(candidate)
    s = validate_and_enrich(prepared, candidate)["segments"][1]
    assert candidate == original  # No silent mutations to raw model choices.
    assert s["instruction"]["available_at_step"] == 10
    assert s["instruction"]["label_origin"] == "retrospective"
    assert s["information_flags"] == ["retrospective_instruction"]
    assert not s["needs_review"]
    assert s["training_eligibility"] == {"low_level": True, "high_level": False}


def test_instruction_time_before_segment_start_is_allowed(episode, candidate):
    candidate["segments"][1]["instruction"]["source_refs"] = ["d0000.assessment.current_subgoal"]
    s = validate_and_enrich(prepare(episode), candidate)["segments"][1]
    assert s["instruction"]["available_at_step"] == 0
    assert s["instruction"]["label_origin"] == "online_supported"


def test_same_sim_step_later_zero_decision_is_not_online(episode, candidate):
    prepared = prepare(episode)
    prepared["payload"]["refs"]["future"] = dict(decision_id=3, available_at_step=4,
                                                  phase="before", field="assessment.current_subgoal", value="Grasp 5")
    candidate["segments"][1]["instruction"]["source_refs"] = ["future"]
    s = validate_and_enrich(prepared, candidate)["segments"][1]
    assert s["instruction"]["available_at_step"] == 4
    assert s["instruction"]["label_origin"] == "retrospective"


def test_all_errors_are_reported_and_drafts_survive(episode, candidate, tmp_path):
    broken = deepcopy(candidate)
    broken["segments"][0]["outcome_evidence_refs"].append("missing_first")
    broken["segments"][1]["outcome_evidence_refs"] = ["missing"]
    errors = collect_errors(prepare(episode), broken)
    assert {e["segment_id"] for e in errors} == {"s1", "s2"}
    class Fake:
        def complete(self, *args):
            return {"text": json.dumps(broken)}
    cfg, out = load_config(), tmp_path / "failed"
    cfg["max_repairs"] = 0
    with pytest.raises(ValueError):
        annotate_one(prepare(episode), out, cfg, protocol(), Fake())
    assert read(out / "draft_00.json") == broken
    assert read(out / "validation_00.json")["errors"] == errors
    assert not (out / "annotation.json").exists()
    assert not (out / "candidate.json").exists()


def test_repair_receives_all_errors_and_preserves_audit(episode, candidate, tmp_path):
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            draft = deepcopy(candidate)
            if self.calls == 1:
                draft["segments"][0]["outcome_evidence_refs"].append("missing_first")
                draft["segments"][1]["outcome_evidence_refs"] = ["bad_ref"]
            else:
                assert {e["segment_id"] for e in correction["errors"]} == {"s1", "s2"}
                assert correction["previous"]["segments"][1]["outcome_evidence_refs"] == ["bad_ref"]
                draft = patch_for(candidate, ["s1", "s2"])
            return {"text": json.dumps(draft)}
    out = tmp_path / "fixed"
    result, _ = annotate_one(prepare(episode), out, load_config(), protocol(), Fake())
    assert result["episode_validation"] == "passed"
    changes = read(out / "repair_changes_01.json")
    assert len(changes) == 2
    assert all(not c["boundaries_changed"] and c["fields"] == ["outcome_evidence_refs"] for c in changes)
    assert read(out / "validation_01.json") == {"valid": True, "errors": []}


def test_no_segments_indexed_when_one_segment_fails(episode, candidate, tmp_path, monkeypatch):
    root = tmp_path / "batch-invalid"
    prepare_batch([episode], root)
    candidate["segments"][1]["outcome_evidence_refs"] = ["nonexistent"]
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions", max_repairs=0)
    monkeypatch.setattr(APIClient, "complete", lambda *args: dict(
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(candidate)))]))
    result = run_batch(root, cfg)
    assert result["failures"] == 1
    assert read(Path(result["run_root"]) / "segments.json") == []


def test_context_includes_all_same_state_reasoning(episode, candidate):
    prepared = prepare(episode)
    context = endpoint_windows(prepared["payload"])["4"]
    assert context["last_execution_decision_id"] == 0
    assert context["reasoning_decision_ids"] == [1, 2]
    assert context["next_execution_decision_id"] == 2
    assert context["state_ref_ids"] == ["d0001.assessment.execution_evidence", "d0002.assessment.execution_evidence"]
    assert not context["exclusive"]
    assert "d0002.motion_result.status" not in context["context_ref_ids"]
    assert validate_and_enrich(prepared, candidate)["segments"][0]["outcome_evidence_context"] == context


def test_same_step_earlier_report_is_usable(episode, candidate):
    candidate["segments"][0]["outcome_evidence_refs"] = ["d0001.assessment.execution_evidence"]
    assert not collect_errors(prepare(episode), candidate)


def test_later_feedback_keeps_timing_for_causal_audit(episode, candidate):
    candidate["segments"][0]["outcome_evidence_refs"] = ["d0003.assessment.execution_evidence"]
    candidate["segments"][0]["outcome_basis"] = "inferred"
    result = validate_and_enrich(prepare(episode), candidate)
    # Structural validation deliberately does not certify semantic attribution.
    assert result["segments"][0]["evidence_timing"][0]["relation"] == "later_feedback"


def test_latest_missing_report_does_not_hide_same_state_observation(episode, candidate):
    prepared = prepare(episode)
    del prepared["payload"]["refs"]["d0002.assessment.execution_evidence"]
    context = endpoint_windows(prepared["payload"])["4"]
    assert context["state_ref_ids"] == ["d0001.assessment.execution_evidence"]
    candidate["segments"][0]["outcome_evidence_refs"] = context["state_ref_ids"]
    assert not collect_errors(prepared, candidate)


def test_terminal_no_feedback_can_remain_justified_unknown(episode, candidate):
    prepared = prepare(episode)
    prepared["payload"]["decisions"] = prepared["payload"]["decisions"][:-1]
    prepared["payload"]["refs"] = {k:v for k,v in prepared["payload"]["refs"].items() if v["decision_id"] != 3}
    context = endpoint_windows(prepared["payload"])["10"]
    assert context["reasoning_decision_ids"] == [] and context["state_ref_ids"] == []
    candidate["segments"][1].update(outcome="unknown", outcome_basis="unresolved",
        unknown_reason="The final receipt does not establish retained possession", outcome_evidence_refs=[])
    assert validate_and_enrich(prepared, candidate)["segments"][1]["assessed_at_step"] is None


def test_receipts_are_bound_to_last_execution_not_zero_step_commands(episode, candidate):
    prepared = prepare(episode)
    windows = endpoint_windows(prepared["payload"])
    assert "d0001.motion_result.error" not in windows["4"]["context_ref_ids"]
    assert "d0002.motion_result.error" in windows["10"]["context_ref_ids"]
    candidate["segments"][1].update(outcome="interrupted", outcome_evidence_refs=["d0002.motion_result.error"])
    assert validate_and_enrich(prepared, candidate)["segments"][1]["outcome"] == "interrupted"


def test_completion_history_can_support_result_subject_to_attribution_audit(episode, candidate):
    prepared = prepare(episode)
    key = "d0003.assessment.task_progress.verified_completed.0"
    prepared["payload"]["refs"][key] = dict(decision_id=3, available_at_step=10, phase="before",
        field="assessment.task_progress.verified_completed.0", value="Object acquired in the second attempt")
    candidate["segments"][1].update(outcome_basis="inferred", outcome_evidence_refs=[key])
    assert validate_and_enrich(prepared, candidate)["segments"][1]["training_eligible"]


def test_model_and_validator_share_windows_without_changing_frozen_input(episode, candidate, tmp_path):
    prepared = prepare(episode)
    original = deepcopy(prepared)
    payload = annotation_payload(prepared)
    assert payload["endpoint_feedback_contexts"] == endpoint_windows(prepared["payload"])
    assert prepared == original
    class Fake:
        def complete(self, prompt, actual, correction=None):
            assert actual["endpoint_feedback_contexts"] == payload["endpoint_feedback_contexts"]
            return {"text": json.dumps(candidate)}
    destination = tmp_path / "window-audit"
    annotate_one(prepared, destination, load_config(), protocol(), Fake())
    assert read(destination / "endpoint_feedback_contexts.json") == payload["endpoint_feedback_contexts"]


def patch_for(candidate, ids):
    return dict(patches=[dict(remove_segment_ids=ids,
                             replacement_segments=[deepcopy(s) for s in candidate["segments"] if s["segment_id"] in ids],
                             root_cause="evidence_selection", explanation="Use direct endpoint evidence.")])


def test_repair_context_contains_verbatim_evidence_and_neighbors(episode, candidate):
    broken = deepcopy(candidate)
    broken["segments"][1]["outcome_evidence_refs"] = ["nonexistent"]
    prepared = prepare(episode)
    correction = build_correction(prepared, broken, collect_errors(prepared, broken))
    assert correction["mode"] == "patch"
    group = correction["repair_groups"][0]
    assert len(group["issues"]) >= 2
    assert group["neighbor_segments"][0]["segment_id"] == "s1"
    assert any(e["role"] == "state_report" and e["value"] == "5 moved with the left gripper during lift"
               for e in group["endpoint_evidence"])
    assert group["boundary_decisions"]


def test_patch_locks_untouched_segments_and_preserves_ranges(candidate):
    before = deepcopy(candidate)
    patch = patch_for(candidate, ["s2"])
    patched = merge_patch(candidate, patch, ["s2"])
    assert patched == candidate == before
    with pytest.raises(ValueError, match="locked"):
        merge_patch(candidate, patch_for(candidate, ["s1"]), ["s2"])
    patch["patches"][0]["replacement_segments"][0]["start_step"] = 0
    with pytest.raises(ValueError, match="outer boundaries"):
        merge_patch(candidate, patch, ["s2"])
    with pytest.raises(ValueError, match="schema"):
        merge_patch(candidate, candidate, ["s1", "s2"])


def test_patch_supports_merge_and_rejects_duplicate_targets(candidate):
    patch = patch_for(candidate, ["s1", "s2"])
    merged = deepcopy(candidate["segments"][0])
    merged["end_step"] = 10
    patch["patches"][0]["replacement_segments"] = [merged]
    assert merge_patch(candidate, patch, ["s1", "s2"])["segments"] == [merged]
    patch["patches"].append(deepcopy(patch["patches"][0]))
    with pytest.raises(ValueError, match="already replaced"):
        merge_patch(candidate, patch, ["s1", "s2"])


def test_nonlocal_structure_uses_full_repair(episode, candidate):
    candidate["segments"][1]["start_step"] = 5
    prepared = prepare(episode)
    correction = build_correction(prepared, candidate, collect_errors(prepared, candidate))
    assert correction["mode"] == "full"


def test_pi05_exception_quarantines_invalid_claim_and_zero_step_gpt(episode, candidate):
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    candidate["segments"][1].update(retry_of=None, recovery_of=None)
    prepared = prepare(episode)
    annotation = validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True)
    first, second = annotation["segments"]
    assert annotation["episode_validation"] == "passed_with_pi05_exclusions"
    assert first["annotation_status"] == "excluded_pi05"
    assert first["raw_annotation"] == candidate["segments"][0]
    assert first["instruction"] is None and first["outcome"] == "unknown"
    assert first["control_step_counts"] == {"pi05": 4}
    assert first["training_eligibility"] == {"low_level": False, "high_level": False}
    assert second["training_eligible"]
    assert annotation["excluded_segment_ids"] == ["s1"]


def test_pi05_exception_rejects_gpt_dependency(episode, candidate):
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    with pytest.raises(AnnotationValidationError) as exc:
        validate_and_enrich(prepare(episode), candidate, allow_pi05_exclusions=True)
    assert any(e["code"] == "excluded_dependency" and e["segment_id"] == "s2" for e in exc.value.errors)


@pytest.mark.parametrize("control", ["gpt", "unknown"])
def test_pi05_exception_cannot_hide_other_controls(episode, candidate, control):
    prepared = prepare(episode)
    prepared["payload"]["decisions"][0]["control_type"] = control
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    with pytest.raises(AnnotationValidationError):
        validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True)


def test_pi05_exception_cannot_hide_mixed_segment_or_structure(episode, candidate):
    prepared = prepare(episode)
    mixed = dict(segments=[deepcopy(candidate["segments"][0])])
    mixed["segments"][0].update(end_step=10, outcome_evidence_refs=["missing"])
    with pytest.raises(AnnotationValidationError):
        validate_and_enrich(prepared, mixed, allow_pi05_exclusions=True)
    prepared["payload"]["decisions"][2]["control_type"] = "pi05"
    candidate["segments"][0]["end_step"] = 5
    with pytest.raises(AnnotationValidationError):
        validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True)


def test_excluded_pi05_never_enters_index_and_cache_is_validated(episode, candidate, tmp_path, monkeypatch):
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    candidate["segments"][1].update(retry_of=None, recovery_of=None)
    root = tmp_path / "batch-exclusion"
    prepare_batch([episode], root)
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions", max_repairs=0)
    monkeypatch.setattr(APIClient, "complete", lambda *args: dict(
        choices=[dict(finish_reason="stop", message=dict(content=json.dumps(candidate)))]))
    summary = run_batch(root, cfg)
    assert summary["failures"] == 0
    assert summary["results"][0]["state"] == "annotated_with_exclusions"
    assert summary["results"][0]["excluded_pi05"] == 1
    index = read(Path(summary["run_root"]) / "segments.json")
    assert [s["segment_id"] for s in index] == ["s2"]
    assert run_batch(root, cfg)["results"][0]["state"] == "reused"


def test_exclusion_occurs_only_after_attempting_repair(episode, candidate, tmp_path):
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    candidate["segments"][1].update(retry_of=None, recovery_of=None)
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            return {"text": json.dumps(candidate if correction is None else {"patches": []})}
    client = Fake()
    out = tmp_path / "excluded-after-repair"
    annotation, state = annotate_one(prepare(episode), out, load_config(), protocol(), client)
    assert client.calls == 4 and state == "annotated_with_exclusions"
    assert read(out / "patch_01.json") == {"patches": []}
    assert read(out / "repair_request_01.json")["repair_groups"]
    assert annotation["excluded_segment_ids"] == ["s1"]


def test_entity_schema_enforces_format_and_local_pi05_error_can_be_excluded(episode, candidate):
    import jsonschema
    from data_processing.subtasks.contract import SCHEMA
    candidate["segments"][0]["instruction"]["object_ref"] = "Object with spaces"
    assert list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(candidate))
    candidate["segments"][1].update(retry_of=None, recovery_of=None)
    result = validate_and_enrich(prepare(episode), candidate, allow_pi05_exclusions=True)
    assert result["excluded_segment_ids"] == ["s1"]


def test_patch_split_requires_complete_coverage_and_updated_links(episode, candidate):
    patch = patch_for(candidate, ["s1", "s2"])
    replacement = deepcopy(candidate["segments"])
    replacement[0]["segment_id"] = "split_a"
    patch["patches"][0]["replacement_segments"] = replacement
    merged = merge_patch(candidate, patch, ["s1", "s2"])
    errors = collect_errors(prepare(episode), merged)
    assert any(e["code"] == "invalid_attempt_link" for e in errors)
    replacement[1].update(retry_of="split_a", recovery_of="split_a")
    assert not collect_errors(prepare(episode), merge_patch(candidate, patch, ["s1", "s2"]))
    replacement[1]["start_step"] = 5
    assert any(e["code"] == "partition" for e in collect_errors(
        prepare(episode), merge_patch(candidate, patch, ["s1", "s2"])))


def test_pi05_dependency_cascades_without_training(episode, candidate):
    prepared = prepare(episode)
    prepared["payload"]["decisions"][2]["control_type"] = "pi05"
    candidate["segments"][0]["outcome_evidence_refs"] = ["missing"]
    result = validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True)
    assert result["excluded_segment_ids"] == ["s1", "s2"]
    assert all(not s["training_eligible"] for s in result["segments"])
    assert result["segments"][1]["validation_errors"][0]["code"] == "excluded_dependency"


def test_patch_schema_is_used_by_api(monkeypatch, episode, candidate):
    monkeypatch.setenv("SUBTASK_API_KEY", "test-secret")
    prepared = prepare(episode)
    broken = deepcopy(candidate)
    broken["segments"][1]["outcome_evidence_refs"] = []
    correction = build_correction(prepared, broken, collect_errors(prepared, broken))
    def handler(request):
        body = json.loads(request.content)
        assert body["text"]["format"]["schema"] == PATCH_SCHEMA
        assert "Repair request" in body["input"][1]["content"]
        return httpx.Response(200, json=dict(status="completed", output=[dict(type="message", content=[
            dict(type="output_text", text=json.dumps(patch_for(candidate, ["s2"])))])]))
    raw = APIClient(load_config(), httpx.MockTransport(handler)).complete("prompt", {}, correction)
    assert "patches" in extract_text(raw, "responses")


def test_repair_noop_cannot_hide_gpt_error_and_bad_patch_is_saved(episode, candidate, tmp_path):
    candidate["segments"][1]["outcome_evidence_refs"] = ["missing"]
    class Fake:
        def complete(self, prompt, payload, correction=None):
            # Returning a full draft in patch mode must never silently replace locked data.
            return {"text": json.dumps(candidate)}
    out = tmp_path / "bad-patch"
    with pytest.raises(ValueError, match="patch violates schema"):
        annotate_one(prepare(episode), out, load_config(), protocol(), Fake())
    assert read(out / "patch_01.json") == candidate
    assert read(out / "validation_01.json")["errors"][0]["code"] == "invalid_patch"
    assert not (out / "annotation.json").exists()


def test_malformed_relation_uses_full_repair_without_context_crash(episode, candidate):
    candidate["segments"][0]["retry_of"] = {"invalid": "relation"}
    prepared = prepare(episode)
    correction = build_correction(prepared, candidate, collect_errors(prepared, candidate))
    assert correction["mode"] == "full"


def test_schema_compatibility_and_nonempty_episode_constraint():
    import jsonschema
    from data_processing.subtasks.contract import SCHEMA
    from data_processing.subtasks.repair import response_schema
    for correction in (None, {"mode": "full"}):
        schema = response_schema(correction, {"total_steps": 1500})
        assert schema["properties"]["segments"]["minItems"] == 1
        assert list(jsonschema.Draft202012Validator(schema).iter_errors({"segments": []}))
        empty_schema = response_schema(correction, {"total_steps": 0})
        assert not list(jsonschema.Draft202012Validator(empty_schema).iter_errors({"segments": []}))
    assert "minItems" not in SCHEMA["properties"]["segments"]  # No global mutation.
    patch = response_schema({"mode": "patch"}, {"total_steps": 1500})
    assert "uniqueItems" not in json.dumps(patch)
    assert "segments" not in patch["properties"]


def test_duplicate_patch_ids_still_rejected_without_uniqueitems(candidate):
    patch = patch_for(candidate, ["s1"])
    patch["patches"][0]["remove_segment_ids"] = ["s1", "s1"]
    with pytest.raises(ValueError, match="must not contain duplicates"):
        merge_patch(candidate, patch, ["s1"])


def test_empty_nonzero_annotation_repaired_in_full_mode(episode, candidate, tmp_path):
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            if self.calls == 1:
                return {"text": '{"segments":[]}'}
            assert correction["mode"] == "full"
            assert any(e["code"] == "incomplete_coverage" for e in correction["errors"])
            return {"text": json.dumps(candidate)}
    client = Fake()
    result, state = annotate_one(prepare(episode), tmp_path / "empty-repair", load_config(), protocol(), client)
    assert client.calls == 2 and state == "annotated"
    assert len(result["segments"]) == 2


def test_three_repair_budget_succeeds_on_last_attempt(episode, candidate, tmp_path):
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            if self.calls == 1:
                broken = deepcopy(candidate)
                broken["segments"][1]["outcome_evidence_refs"] = ["missing"]
                return {"text": json.dumps(broken)}
            assert correction["mode"] == "patch"
            return {"text": json.dumps(patch_for(candidate, ["s2"]) if self.calls == 4 else {"patches": []})}
    cfg = load_config()
    assert cfg["max_repairs"] == 3
    client, out = Fake(), tmp_path / "three-repairs"
    _, state = annotate_one(prepare(episode), out, cfg, protocol(), client)
    assert state == "annotated" and client.calls == 4
    assert read(out / "validation_03.json")["valid"]
    assert annotate_one(prepare(episode), out, cfg, protocol(), client)[1] == "reused"
    assert client.calls == 4


def test_three_repair_budget_stops_after_four_calls(episode, candidate, tmp_path):
    class Fake:
        calls = 0
        def complete(self, prompt, payload, correction=None):
            self.calls += 1
            broken = deepcopy(candidate)
            broken["segments"][1]["outcome_evidence_refs"] = ["missing"]
            return {"text": json.dumps(broken if correction is None else {"patches": []})}
    client = Fake()
    with pytest.raises(ValueError, match="after 4 attempts"):
        annotate_one(prepare(episode), tmp_path / "exhausted", load_config(), protocol(), client)
    assert client.calls == 4


def test_codex_failure_prefers_fatal_events_over_warning_and_stderr(tmp_path):
    from data_processing.subtasks.client import codex_failure_detail
    message = json.dumps(dict(status=400, error=dict(code="invalid_json_schema", message="uniqueItems not permitted")))
    events = [dict(type="item.completed", item=dict(type="error", message="Under-development warning")),
              dict(type="error", message=message), dict(type="turn.failed", error=dict(message=message))]
    (tmp_path / "events.jsonl").write_text('\n'.join(json.dumps(e) for e in events) + '\nmalformed line')
    (tmp_path / "stderr.log").write_text("")
    detail = codex_failure_detail(tmp_path)
    assert "400: invalid_json_schema: uniqueItems not permitted" in detail
    assert "events.jsonl" in detail and "stderr.log" in detail
    assert "Under-development" not in detail


def test_codex_failure_fallback_and_secret_redaction(tmp_path):
    from data_processing.subtasks.client import codex_failure_detail
    (tmp_path / "stderr.log").write_text("network failed; Bearer secret-token sk-secret123 https://x/?access_token=secret2")
    detail = codex_failure_detail(tmp_path)
    assert "network failed" in detail
    assert "secret-token" not in detail and "sk-secret123" not in detail and "secret2" not in detail
    (tmp_path / "stderr.log").write_text("")
    assert "No fatal event" in codex_failure_detail(tmp_path)


def test_codex_exit_reports_event_error_and_sends_episode_schema(monkeypatch, tmp_path):
    from hybrid_rollout.robodojo.codex_backend import profiles
    monkeypatch.setattr(profiles, "account_home", lambda *args: tmp_path / "managed-home")
    class FakeProcess:
        returncode = 1
        def __init__(self, command, **kwargs):
            schema = read(Path(command[command.index("--output-schema") + 1]))
            assert schema["properties"]["segments"]["minItems"] == 1
            kwargs["stdout"].write(json.dumps(dict(type="turn.failed", error=dict(message="upstream schema rejected"))) + '\n')
        def communicate(self, value, timeout):
            pass
    monkeypatch.setattr("data_processing.subtasks.client.subprocess.Popen", FakeProcess)
    with pytest.raises(ValueError, match="upstream schema rejected") as exc:
        CodexClient(load_config(), "codex_c", 99, tmp_path / "runtime").complete("prompt", {"total_steps": 1})
    assert "events.jsonl" in str(exc.value)


def test_api_sends_nonempty_full_repair_schema(monkeypatch):
    monkeypatch.setenv("SUBTASK_API_KEY", "test-secret")
    def handler(request):
        body = json.loads(request.content)
        assert body["text"]["format"]["schema"]["properties"]["segments"]["minItems"] == 1
        return httpx.Response(200, json={})
    APIClient(load_config(), httpx.MockTransport(handler)).complete("prompt", {"total_steps": 1}, {"mode": "full"})


@pytest.fixture(autouse=True)
def confirmed_audit_for_existing_stage_tests(monkeypatch):
    from data_processing.subtasks.audit import apply_reviews
    def audit(prepared, candidate, annotation, destination, cfg, spec, client, repairs_remaining):
        report = dict(reviews=[dict(segment_id=s["segment_id"], verdict="confirmed", explanation="Synthetic fixture verified")
                              for s in annotation["segments"] if s["annotation_status"] == "validated"])
        write(destination / "audit_final.json", report)
        return candidate, apply_reviews(annotation, report)
    monkeypatch.setattr("data_processing.subtasks.pipeline.audit_semantics", audit)

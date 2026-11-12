"""Core-goal labels through validation/repair/audit; model responses are fixtures."""
from copy import deepcopy
import json

import pytest

from data_processing.tests.test_subtasks import episode, candidate, patch_for
from data_processing.tests.test_subtasks_v6 import reviews_for
from data_processing.subtasks.audit import validate_final
from data_processing.subtasks.io import read, write
from data_processing.subtasks.pipeline import annotate_one, load_config, protocol
from data_processing.subtasks.prepare import prepare
from data_processing.subtasks.repair import CORE_OUTCOME_POLICY


def test_key_skill_success_keeps_full_interval_without_auxiliary_outcome(episode, candidate):
    path = episode / "controller/history.json"
    history = read(path)
    history[2]["assessment"].update(
        current_subgoal="Left grasp digit 5; right retracts as an auxiliary motion")
    history[3]["assessment"]["execution_evidence"] = (
        "5 moved with the left gripper during lift; right-arm clearance is unconfirmed.")
    write(path, history)
    result = validate_final(prepare(episode), candidate, dict(reviews=reviews_for(candidate)))
    s = result["segments"][1]
    assert s["instruction"]["text"] == "Left arm: Grasp digit 5."
    assert s["outcome"] == "success" and s["training_eligible"]
    assert s["control_step_counts"] == {"gpt": 6}
    assert (s["start_step"], s["end_step"]) == (4, 10)
    assert not s["review_reasons"]
    assert "component_outcomes" not in s


@pytest.mark.parametrize("arm", ["left", "both"])
def test_downstream_inference_survives_structural_and_audit_retries(
        episode, candidate, tmp_path, arm):
    path = episode / "controller/history.json"
    history = read(path)
    goal = ("Move and turn digit 5 above support" if arm == "left" else
            "Left moves digit 5 above support while right positions the support for placement")
    history[0]["assessment"]["current_subgoal"] = goal
    for d in history[1:3]:
        d["assessment"].update(current_subgoal="Place digit 5 on support",
                               execution_evidence="Proceed to release using the resulting pose.")
    history[3]["assessment"]["execution_evidence"] = (
        "5 was released onto the positioned support in the intended placement pose, without readjustment.")
    write(path, history)
    first, second = candidate["segments"]
    first["instruction"].update(skill_type="custom", arm=arm, custom_description=goal,
                                 destination_ref="support", target_relation="to")
    first.update(outcome="success", outcome_basis="inferred",
                 outcome_evidence_refs=["d0003.assessment.execution_evidence"],
                 outcome_rationale="The following placement used the resulting pose without correction, "
                 "supporting completion of the preceding preparation goal.")
    second["instruction"].update(skill_type="place", destination_ref="support", target_relation="on")
    second.update(retry_of=None, recovery_of=None, purpose="task")
    initial = deepcopy(candidate)
    initial["segments"][0].update(outcome="unknown", outcome_basis="unresolved",
        unknown_reason="No separate component completion statements.",
        review_reasons=["concurrent_subtasks"])
    # Force a structural retry, then a rejected audit, then a corrected audit.
    class Fake:
        modes = []
        audits = 0

        def complete(self, prompt, payload, correction=None):
            self.modes.append(correction["mode"] if correction else "initial")
            if correction is None:
                invalid = deepcopy(initial)
                invalid["segments"][0]["outcome_evidence_refs"] = ["missing"]
                return {"text": json.dumps(invalid)}
            assert CORE_OUTCOME_POLICY in correction["repair_policy"]
            if correction["mode"] == "patch":
                return {"text": json.dumps(patch_for(initial, ["s1"]))}
            self.audits += 1
            if self.audits == 1:
                return {"text": json.dumps(dict(patches=[], reviews=[]))}
            assert correction["errors"] and correction["previous"] == initial
            return {"text": json.dumps(dict(**patch_for(candidate, ["s1"]),
                                             reviews=reviews_for(candidate)))}

    client = Fake()
    destination = tmp_path / arm
    result, state = annotate_one(prepare(episode), destination, load_config(), protocol(), client)
    assert state == "annotated"
    assert client.modes == ["initial", "patch", "audit", "audit"]
    s = result["segments"][0]
    assert result["version"] == "subtasks.v9" and s["criterion_id"] == "custom.v4"
    assert s["outcome"] == "success" and s["outcome_basis"] == "inferred"
    assert s["training_eligibility"] == dict(low_level=True, high_level=True)
    assert not s["review_reasons"]
    assert s["evidence_timing"][0]["relation"] == "later_feedback"
    assert s["instruction"]["label_origin"] == "online_supported"
    assert read(destination / "candidate.json") == candidate

"""Policy-visible task boundaries, reference transport and deployed assets."""
import copy
import json
from pathlib import Path
import re

from PIL import Image
import pytest

from .task_success_conditions import render_task_success_conditions
from . import task_references
from .skill_auto_infer.run import content_items
from .skill_auto_infer.subtask_instruction import render_instruction, schema
from .io import InputError

TASKS = (
    "classify_objects", "classify_objects_by_language", "fill_pen_holder",
    "imitate_sorting_sequence", "organize_table", "stack_blocks_by_language",
    "arrange_largest_number", "pack_objects_into_box", "sort_nesting_dolls_by_size",
    "store_tools_in_toolbox", "pick_from_conveyor_by_image", "sweep_blocks",
)


@pytest.mark.parametrize("task", TASKS + tuple(
    name + "_random" for name in (
        "arrange_largest_number", "pack_objects_into_box", "sort_nesting_dolls_by_size", "sweep_blocks")))
def test_selected_task_has_no_hidden_measurement_requirements(task):
    text = render_task_success_conditions(task)
    assert not re.search(
        r"\b(?:world|local)\s+[+-]?[xyz]\b|\b(?:root|checkpoint|functional|cat[0-9])\b"
        r"|\bXY\b|\d+(?:\.\d+)?\s*(?:m\b|cm\b|°)|categories \d|center_flip",
        text)
    assert text.count("## ") == 1
    if "sort_nesting" in task:
        assert "smallest to largest" in text
        assert "largest to smallest" not in text
    if "arrange_largest" not in task:
        assert "Flip recovery:" not in text


def test_pending_reference_is_explicit_and_other_tasks_do_not_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr(task_references, "REFERENCE_ROOT", tmp_path / "missing")
    monkeypatch.delenv("ROLLOUT_TASK_REFERENCE_IMAGES", raising=False)
    assert task_references.prepare_task_references("store_tools_in_toolbox", tmp_path) == []
    manifest = json.loads((tmp_path / "task_references/manifest.json").read_text())
    assert manifest["status"] == "pending_image"
    assert task_references.prepare_task_references("classify_objects", tmp_path / "other") == []
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("evaluation", [False, True])
def test_fixed_image_delivered_separately_without_mutating_live_or_policy_data(
        tmp_path, monkeypatch, evaluation):
    from . import regression_eval
    monkeypatch.setattr(regression_eval, "enabled", lambda *args: evaluation)
    monkeypatch.delenv("ROLLOUT_TASK_REFERENCE_IMAGES", raising=False)
    root = tmp_path / "assets"
    directory = root / "store_tools_in_toolbox"
    directory.mkdir(parents=True)
    Image.new("RGB", (80, 40), "red").save(directory / "completed.png")
    monkeypatch.setattr(task_references, "REFERENCE_ROOT", root)
    refs = task_references.prepare_task_references("store_tools_in_toolbox", tmp_path / "audit")
    Image.new("RGB", (40, 40), "blue").save(tmp_path / "live.png")
    packet = dict(images=[dict(camera="cam_high", path=str(tmp_path / "live.png"))],
                  instruction="Place tools.", current_state=[0] * 14)
    original = copy.deepcopy(packet)
    result = content_items(packet, images=True, max_image_edge=32, task_references=refs)
    visible = json.loads(result[0]["text"])
    assert packet == original
    assert len(visible["images"]) == 1 and visible["images"][0]["camera"] == "cam_high"
    ref = visible["task_reference_images"][0]
    assert ref["current_scene"] is False and ref["role"] == "fixed_completed_state_example"
    assert ref["codex_preview"]["size"] == [32, 16]
    assert len(result) == 3
    assert "task_reference_images" not in json.loads(
        content_items(packet, images=False, task_references=refs)[0]["text"])
    monkeypatch.setenv("ROLLOUT_TASK_REFERENCE_IMAGES", "0")
    assert task_references.prepare_task_references("store_tools_in_toolbox", tmp_path / "off") == []


def test_reference_png_is_in_deployment_identity(tmp_path):
    from .deployment import source_files, source_identity
    root = tmp_path / "hybrid_rollout/robodojo/task_references/store_tools_in_toolbox"
    root.mkdir(parents=True)
    image = root / "completed.png"
    Image.new("RGB", (8, 8), "red").save(image)
    assert image in [p for p, _ in source_files(tmp_path)]
    before = source_identity(tmp_path)
    Image.new("RGB", (8, 8), "blue").save(image)
    assert source_identity(tmp_path) != before


def test_push_release_schema_and_legacy_text_compatibility():
    from data_processing.subtasks.contract import canonical, SKILLS
    i = dict(skill_type="push", arm="both", object_ref="keyboard",
             destination_ref="frame", target_relation="to", target_orientation_ref=None,
             custom_description=None, giver_arm="unknown", receiver_arm="unknown")
    assert render_instruction(i) == canonical(i) == "Both arms: Push keyboard to frame."
    i.update(skill_type="release", destination_ref=None, target_relation="none", arm="left")
    assert render_instruction(i) == canonical(i) == "Left arm: Release keyboard."
    assert "release" in SKILLS and "reposition" not in SKILLS
    assert "reposition" not in schema()["anyOf"][0]["properties"]["skill_type"]["enum"]
    with pytest.raises(InputError):
        render_instruction(dict(i, destination_ref="frame", target_relation="on"))
    legacy = dict(i, skill_type="reposition", destination_ref="frame", target_relation="to")
    assert canonical(legacy) == "Left arm: Reposition keyboard to frame."
    with pytest.raises(InputError):
        render_instruction(legacy)

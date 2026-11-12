"""Verify task selection and the actual GPT startup prompt boundary."""

import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from .task_success_conditions import (
    CATALOG,
    GUIDANCE,
    append_task_success_conditions,
    render_task_success_conditions,
)


def test_catalog_covers_standard_and_random_tasks():
    """The reviewed catalog includes all54 runtime names without duplicates."""
    tasks = re.findall(r"^## ([A-Za-z0-9_]+)$", CATALOG.read_text(), re.MULTILINE)
    assert len(tasks) == len(set(tasks)) == 54
    assert sum(task.endswith("_random") for task in tasks) == 12
    for task in tasks:
        section = render_task_success_conditions(task)
        assert re.findall(r"^## ([A-Za-z0-9_]+)$", section, re.MULTILINE) == [task]
        assert "file SHA256" not in section
        assert "query(...,0)" not in section
        assert "inner list" not in section
        assert "Maintainer reference" not in section
        # Digit tasks include the reviewed low-release safety and attempt limits.
        limit = 4000 if task in ('arrange_largest_number', 'arrange_largest_number_random') else 2400
        assert len(section) < limit
        assert not re.search(r"[\u3400-\u9fff]", section)


def test_deposit_coin_rules_include_full_bbox_and_robot_return():
    """Coin insertion requires complete containment and the native robot return."""
    text = render_task_success_conditions("deposit_coin")
    assert "entire bounding-box projection" in text and "bottom" in text and "center" in text
    assert "Return the arms" in text
    assert "recorded starting poses" in text
    assert "20-point process score" not in text
    assert "## insert_tubes" not in text


def test_random_variant_keeps_its_distinct_success_rules():
    """Do not silently map random tasks onto their standard reward definition."""
    standard = render_task_success_conditions("stack_blocks")
    random = render_task_success_conditions("stack_blocks_random")
    assert "0.0175m" in standard
    assert "fixed <=0.015m" in random and "0.0175m" not in random
    assert "all the target blocks" in render_task_success_conditions("sweep_blocks")
    assert "all the target objects" in render_task_success_conditions("sweep_blocks_random")


def test_task_specific_triggers_and_handoff_are_preserved():
    for task in ("arrange_largest_number", "arrange_largest_number_random"):
        text = render_task_success_conditions(task)
        assert "Operation preference, not a success requirement" in text
        assert "Preserve a suitable student staging/handoff plan" in text
        assert "releasing and withdrawing" in text
        assert "Center each digit on its pad" in text
    fold = render_task_success_conditions("fold_clothes")
    assert "returning either arm" in fold and "causes failure" in fold
    assert "(-0.095,0.17)" in fold and ">=0.7" in fold
    assert "handoff" not in fold
    assert "these stages in order" in render_task_success_conditions("cover_blocks")
    assert "more than three counted presses fails" in render_task_success_conditions("swap_blocks")
    assert "within 0.15m" not in render_task_success_conditions("insert_key")
    assert "within 0.15m" not in render_task_success_conditions("general_pickup")


def test_supported_flip_advice_is_digit_only_and_preserves_stable_transport():
    for task in ('arrange_largest_number', 'arrange_largest_number_random'):
        text = render_task_success_conditions(task)
        assert 'Use a securely graspable region for ordinary transport' in text
        assert 'grasping near the head, tail or one side' in text
        assert 'one possible method, not a guarantee' in text
        assert 'Do not mistake ordinary rotation or an ambiguous view for a mirror flip' in text
        assert 'for rotation alone, adjust the in-plane orientation instead' in text
        assert 'An in-plane turn alone cannot correct a true face reversal' not in text
        assert 'release from very low height' in text
        assert 'fresh observations to verify each attempt' in text
        assert 'do not use extreme wrist/arm postures' in text
        assert 'sustained wrist twisting, flinging, or release from high above' in text
        assert '2-3 attempts in total, not a fresh budget on each decision' in text
        assert 'early-abandon tool is available and its gate permits it' in text
        assert 'opposite end supported on the table' not in text
        assert 'regrasp stably for transport' in text
        assert text.count('Flip recovery:') == 1
    tasks = re.findall(r'^## ([A-Za-z0-9_]+)$', CATALOG.read_text(), re.MULTILINE)
    for task in tasks:
        if task not in ('arrange_largest_number', 'arrange_largest_number_random'):
            assert 'Flip recovery:' not in render_task_success_conditions(task)


def test_unknown_task_and_duplicate_catalog_fail_loudly(tmp_path):
    """No substitute task or ambiguous duplicate heading may enter the prompt."""
    with pytest.raises(ValueError, match="No reviewed"):
        render_task_success_conditions("deposit_coin_typo")
    catalog = tmp_path / "duplicate.md"
    catalog.write_text("# Rules\n## task\nA\n## task\nB\n")
    with pytest.raises(ValueError, match="duplicate"):
        render_task_success_conditions("task", catalog=catalog)


def test_append_preserves_prompt_and_audits_exact_section(tmp_path):
    """The existing gate is retained verbatim; repeated injection is rejected."""
    original = "Keep the existing gate and robot safety contract.\n"
    result = append_task_success_conditions(original, "insert_key", output_dir=tmp_path)
    section = (tmp_path / "task_success_conditions.md").read_text()
    assert result == original + "\n\n" + GUIDANCE.read_text() + "\n\n" + section
    assert section == render_task_success_conditions("insert_key")
    with pytest.raises(ValueError, match="already"):
        append_task_success_conditions(result, "deposit_coin")


@pytest.mark.parametrize(
    "route,method,query,motion",
    [
        ("skill", "pi05_plus_gpt", False, False),
        ("skill", "gpt_only", False, False),
        ("skill_auto_infer", "pi05_plus_gpt", False, False),
        ("skill_auto_infer", "pi05_plus_gpt", True, True),
    ],
)
def test_policy_start_sends_rules_to_gpt(
    tmp_path, monkeypatch, route, method, query, motion
):
    """Capture thread/start without opening a model or simulator connection."""
    from importlib import import_module

    module = import_module(f".{route}.run", __package__)
    captured = {}

    class StartupTransport:
        """Only the Codex process boundary is replaced in this startup check."""

        def __init__(self, argv, output):
            self.process = SimpleNamespace(pid=0)

        def request(self, method, params, timeout):
            if method == "initialize":
                return {}
            assert method == "thread/start"
            captured.update(params)
            return {
                "model": module.MODEL,
                "reasoningEffort": module.EFFORT,
                "thread": {"id": "offline-startup-check"},
            }

        def notify(self, method, params):
            assert method == "initialized"

        def close(self):
            pass

    # Skip only the unrelated binary version query; use real prompt composition,
    # workspace creation, settings, tool schemas and thread/start serialization.
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="offline-check")
    )
    monkeypatch.setenv("ROLLOUT_ENABLE_MOTION_TOOLS", "1" if motion else "0")
    kwargs = dict(method=method, task="deposit_coin", transport_factory=StartupTransport)
    if route == "skill_auto_infer":
        kwargs["scene_query_enabled"] = query
    worker = module.CodexPolicy(tmp_path / "controller/codex_workspace", "unused", **kwargs)
    try:
        prompt = captured["developerInstructions"]
        assert prompt == worker.prompt == (worker.workspace / "PROMPT.md").read_text()
        assert prompt.count("# Native task success conditions: deposit_coin") == 1
        assert "The entire bounding-box projection of coin0" in prompt
        metadata = json.loads((worker.workspace / "worker.json").read_text())
        assert metadata["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert not re.search(r"[\u3400-\u9fff]", (worker.workspace / "task_success_conditions.md").read_text())
        assert (worker.workspace / "task_success_conditions.md").read_text() in prompt
        assert metadata["evaluation_method"] == method
        if query:
            assert metadata["scene_query_enabled"] and metadata["motion_tools_enabled"]
    finally:
        worker.close()

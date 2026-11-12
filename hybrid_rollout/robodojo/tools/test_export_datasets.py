import json

import numpy as np
import pytest

from .extract_episode import CAMERAS, extract
from .export_datasets import export, prompt, schemas, sources
from .test_extract_episode import fixture


def prepared(tmp_path):
    raw = fixture(tmp_path)
    s = json.loads((raw / "sim/summary.json").read_text())
    s.update(success=True, evaluation_case=dict(task="test_task"))
    (raw / "sim/summary.json").write_text(json.dumps(s))
    for t in range(3):
        p = raw / "sim/e/observations" / f"{t:06d}.npz"
        with np.load(p) as z:
            values = {k: z[k] for k in z.files}
        values.update(
            {k: np.full((32, 32, 3), t * 50, dtype=np.uint8) for k in CAMERAS}
        )
        np.savez(p, **values)
    out = tmp_path / "extract"
    extract(raw, out)
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(failed=0, results=[dict(output=str(out))])))
    return index, out


def test_both_roundtrip(tmp_path):
    index, out = prepared(tmp_path)
    result = export(index, tmp_path / "exports", "next_state")
    assert result["status"] == "passed" and result["total_steps"] == 2
    assert result["lerobot_validation"]["rows"] == 2
    assert result["episodes"][0]["mcap_validation"]["status"] == "passed"
    with pytest.raises(FileExistsError):
        export(index, tmp_path / "exports", "command")


def test_bad_input_rejected(tmp_path):
    index, out = prepared(tmp_path)
    (out / "steps.npz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        sources(index)


def test_write_only_does_not_claim_readback(tmp_path, monkeypatch):
    index, _ = prepared(tmp_path)
    from . import export_datasets as module
    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected full readback")
    monkeypatch.setattr(module, "verify_mcap", unexpected)
    monkeypatch.setattr(module, "verify_lerobot", unexpected)
    result = export(index, tmp_path / "exports", "command", validation="write_only")
    assert result["status"] == "written"
    assert result["lerobot_validation"]["status"] == "not_run"
    assert result["episodes"][0]["mcap_validation"]["status"] == "not_run"


def test_command_label_and_episode_boundary(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    index, out = prepared(first)
    second = tmp_path / "second"
    second.mkdir()
    raw = fixture(second)
    (raw / "sim/e").rename(raw / "sim/e2")
    s = json.loads((raw / "sim/summary.json").read_text())
    s.update(episode_id="e2", success=True, evaluation_case=dict(task="test_task"))
    (raw / "sim/summary.json").write_text(json.dumps(s))
    for t in range(3):
        p = raw / "sim/e2/observations" / f"{t:06d}.npz"
        with np.load(p) as z:
            values = {k: z[k] for k in z.files}
        values.update(
            {k: np.full((32, 32, 3), 255 - t * 50, dtype=np.uint8) for k in CAMERAS}
        )
        np.savez(p, **values)
    other = second / "extract"
    extract(raw, other)
    index.write_text(
        json.dumps(
            dict(failed=0, results=[dict(output=str(out)), dict(output=str(other))])
        )
    )
    result = export(index, tmp_path / "exports", "command", formats="lerobot")
    assert result["lerobot_validation"]["rows"] == 4
    assert len(result["episodes"]) == 2


def test_prompt_only_uses_contemporaneous_fields():
    e = dict(task_instruction="Sort digits", future_result="success")
    d = dict(
        response=dict(
            assessment=dict(
                current_subgoal="Grasp 8",
                execution_status="failed",
                execution_evidence="Missed",
            )
        )
    )
    assert prompt(e, d, "task") == "Sort digits"
    assert "Missed" in prompt(e, d, "subtask") and "success" not in prompt(
        e, d, "subtask"
    )
    assert "fields" in schemas()["custom.TextMap"]["required"]

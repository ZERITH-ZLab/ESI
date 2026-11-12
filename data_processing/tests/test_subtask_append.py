"""Incremental batches preserve completed labels and never duplicate retries."""
import json
from pathlib import Path
import shutil

import pytest

from data_processing.tests.test_subtasks import episode, candidate
from data_processing.tests.test_subtasks_v6 import reviews_for
from data_processing.subtasks.client import APIClient
from data_processing.subtasks.io import read, write
from data_processing.subtasks.pipeline import batch_lock, load_config, prepare_batch, run_batch
from data_processing.subtasks.prepare import discover


def clone_case(episode):
    other = Path(str(episode).replace("layout_0", "layout_1"))
    shutil.copytree(episode, other)
    write(other.parent / "sample.json", dict(repeat_id=1, attempts=[dict(archive=str(other), finished_at=124)]))
    return other


def test_append_reuses_labels_and_refreshes_combined_index(episode, candidate, tmp_path, monkeypatch):
    source, batch = episode.parents[6], tmp_path / "batch"
    original = prepare_batch([source], batch, latest_per_case=True)
    cfg = load_config()
    cfg.update(backend="api", api_mode="chat_completions")
    calls = []

    def complete(self, prompt, payload, correction=None):
        calls.append(correction)
        output = dict(reviews=reviews_for(candidate), patches=[]) if correction else candidate
        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(output)))])

    monkeypatch.setattr(APIClient, "complete", complete)
    first = run_batch(batch, cfg)
    assert first["failures"] == 0
    old_label = Path(first["results"][0]["annotation"])
    old_bytes, old_mtime = old_label.read_bytes(), old_label.stat().st_mtime_ns
    original_calls = len(calls)
    clone_case(episode)
    assert prepare_batch([source], batch, latest_per_case=True) == original
    expanded = prepare_batch([source], batch, append=True, latest_per_case=True)
    assert len(expanded["episodes"]) == 2
    assert expanded["episodes"][0] == original["episodes"][0]
    assert len(list((batch / "manifest_history").glob("*.json"))) == 1
    assert prepare_batch([source], batch, append=True, latest_per_case=True) == expanded
    second = run_batch(batch, cfg)
    assert second["run_id"] == first["run_id"]
    assert second["failures"] == 0
    assert [row["state"] for row in second["results"]] == ["reused", "annotated"]
    assert len(calls) == 2 * original_calls
    assert old_label.read_bytes() == old_bytes and old_label.stat().st_mtime_ns == old_mtime
    assert len(read(Path(second["run_root"]) / "segments.json")) == 4
    assert len(read(Path(second["run_root"]) / "episodes.index.json")["episodes"]) == 2


def test_latest_attempt_does_not_fall_back_or_replace_frozen_case(episode, tmp_path):
    source, batch = episode.parents[6], tmp_path / "batch"
    prepare_batch([source], batch, latest_per_case=True)
    newer = episode.parent / "attempt_001"
    sample = read(episode.parent / "sample.json")
    sample["attempts"].append(dict(archive=str(newer), finished_at=None))
    write(episode.parent / "sample.json", sample)
    archives, excluded = discover([source], latest_per_case=True)
    assert archives == [] and len(excluded) == 1
    assert discover([source])[0] == [episode]
    shutil.copytree(episode, newer)
    sample["attempts"][-1]["finished_at"] = 125
    write(episode.parent / "sample.json", sample)
    assert discover([source], latest_per_case=True)[0] == [newer]
    previous = (batch / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="different latest attempt"):
        prepare_batch([source], batch, append=True, latest_per_case=True)
    assert (batch / "manifest.json").read_bytes() == previous


def test_append_checks_fingerprints_lock_and_selection(episode, tmp_path):
    source, batch = episode.parents[6], tmp_path / "batch"
    prepare_batch([source], batch)
    with batch_lock(batch), pytest.raises(ValueError, match="owns this output directory"):
        prepare_batch([source], batch, append=True)
    with pytest.raises(ValueError, match="Selection changed"):
        prepare_batch([source], batch, append=True, latest_per_case=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_batch([source], batch, append=True, limit=1)
    (episode / "controller/history.json").write_text("[]")
    with pytest.raises(ValueError, match="Source changed"):
        prepare_batch([source], batch, append=True)

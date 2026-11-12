import json
import sys

import pytest

from . import export_campaign as campaign


@pytest.mark.parametrize("fail_pilot", [False, True])
def test_pilot_gates_parallel_exports(tmp_path, monkeypatch, fail_pilot):
    output = tmp_path / "out"
    pilot_done = []
    calls = []

    def fake(command, log, env=None):
        from pathlib import Path
        dest = Path(command[command.index("--output") + 1])
        dest.mkdir(parents=True)
        if "--experiment" in command:
            rows = []
            for i in range(6):
                p = tmp_path / f"input_{i}"
                p.mkdir()
                (p / "manifest.json").write_text(json.dumps(dict(episode_id=str(i))))
                rows.append(dict(output=str(p)))
            (dest / "index.json").write_text(json.dumps(dict(failed=0, results=rows)))
            return
        validation = command[command.index("--validation") + 1]
        rows = campaign.read(Path(command[command.index("--index") + 1]))["results"]
        assert command[command.index("--action-label") + 1] == "command"
        calls.append(validation)
        if validation == "full":
            if fail_pilot:
                raise ValueError("Pilot failed")
            pilot_done.append(True)
        else:
            assert pilot_done
        from .export_datasets import digest
        (dest / "export_manifest.json").write_text(json.dumps(dict(
            action_label="command", validation=validation,
            input_sha256=digest(Path(command[command.index("--index") + 1])),
            status="passed" if validation == "full" else "written", total_steps=len(rows),
            episodes=[campaign.read(Path(r["output"]) / "manifest.json") for r in rows])))

    monkeypatch.setattr(campaign, "run_child", fake)
    monkeypatch.setattr(sys, "argv", ["export_campaign", "--experiment", str(tmp_path / "raw"),
                                     "--output", str(output), "--episodes", "6", "--shard-size", "2"])
    if fail_pilot:
        with pytest.raises(ValueError, match="Pilot failed"):
            campaign.main()
        assert calls == ["full"]
        assert not (output / "dataset_collection.json").exists()
        assert (output / "failure.json").exists()
    else:
        campaign.main()
        result = campaign.read(output / "dataset_collection.json")
        assert result["total_episodes"] == 6 and result["total_steps"] == 6
        assert calls.count("write_only") == 2
        # A resume reuses completed shards and preserves/restarts only a partial one.
        (output / "shard_002/export_manifest.json").unlink()
        (output / "shard_002/partial.bin").write_bytes(b"preserve")
        calls.clear()
        monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
        campaign.main()
        assert calls == ["write_only"]
        saved = list((output / "interrupted").glob("*/shard_002/partial.bin"))
        assert len(saved) == 1 and saved[0].read_bytes() == b"preserve"

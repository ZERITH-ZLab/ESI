"""Extract a frozen campaign, validate two command-label episodes, export shards.

Each subprocess owns a separate LeRobot root / HF cache. No shared dataset writes.
Collection index is published only when every shard finishes; originals are read-only.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .extract_episode import read, require, write


def emit(event, **kw):
    print(json.dumps(dict(event=event, time=time.time(), **kw)), flush=True)


def run_child(command, log, env=None):
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    require(result.returncode == 0, f"Subprocess exit {result.returncode}: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    require(a.episodes >= 2 and 1 <= a.workers <= 4 and a.shard_size > 0, "Invalid sizes")
    root, source = a.output.resolve(), a.experiment.resolve()
    require(not root.is_relative_to(source), "Output must be outside rollout")
    root.mkdir(parents=True, exist_ok=a.resume)
    # Keep the descriptor alive for the entire campaign. Separate launches cannot
    # rename partial shards while another campaign is still writing them.
    lock = (root / ".campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_id = uuid.uuid4().hex
    logs = root / "logs" / run_id
    logs.mkdir(parents=True)
    (root / "indexes").mkdir(exist_ok=True)
    config = dict(source=str(source), episodes=a.episodes, workers=a.workers,
                  shard_size=a.shard_size, action_label="command", fps_source="control_dt")
    if a.resume:
        require(read(root / "campaign_config.json") == config, "Resume configuration differs")
        for previous in ("dataset_collection.json", "failure.json"):
            if (root / previous).exists():
                (root / previous).rename(logs / previous)
    else:
        write(root / "campaign_config.json", config)
    try:
        if not (a.resume and (root / "extracted/index.json").exists()):
            emit("extract_start", **config)
            run_child([sys.executable, "-u", "-m", "hybrid_rollout.robodojo.tools.extract_batch",
                       "--experiment", str(source), "--output", str(root / "extracted"),
                       "--limit", str(a.episodes), "--workers", str(a.workers)], logs / "extract.log")
        extracted = read(root / "extracted/index.json")
        require(extracted["failed"] == 0 and len(extracted["results"]) == a.episodes,
                "Incomplete extraction")
        rows = extracted["results"]
        groups = [rows[:2]] + [rows[i:i+a.shard_size] for i in range(2, len(rows), a.shard_size)]

        def shard(i):
            name = f"shard_{i:03d}"
            index = root / "indexes" / f"{name}.json"
            expected_index = dict(failed=0, results=groups[i])
            if index.exists():
                require(read(index) == expected_index, f"Shard selection changed: {name}")
            else:
                write(index, expected_index)
            destination = root / name
            report_path = destination / "export_manifest.json"
            validation = "full" if i == 0 else "write_only"
            reused = a.resume and report_path.exists()
            if reused:
                report = read(report_path)
                require(report["action_label"] == "command" and report["validation"] == validation,
                        f"Shard export contract differs: {name}")
                # Small identity checks only, not another RGB/trajectory readback.
                expected_ids = [read(Path(row["output"]) / "manifest.json")["episode_id"] for row in groups[i]]
                require([e["episode_id"] for e in report["episodes"]] == expected_ids,
                        f"Shard episode identity differs: {name}")
                from .export_datasets import digest
                require(report["input_sha256"] == digest(index), f"Shard index checksum changed: {name}")
                emit("shard_reused", shard=name)
            elif destination.exists():
                require(a.resume, f"Shard already exists: {name}")
                preserved = root / "interrupted" / run_id / name
                preserved.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(preserved)
                emit("partial_shard_preserved", shard=name, path=str(preserved))
            env = dict(os.environ, HF_DATASETS_CACHE=str(root / "cache" / name),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            if not reused:
                emit("shard_start", shard=name, episodes=len(groups[i]), validation=validation)
                run_child([sys.executable, "-u", "-m", "hybrid_rollout.robodojo.tools.export_datasets",
                           "--index", str(index), "--output", str(destination),
                           "--action-label", "command", "--formats", "both", "--validation",
                           validation], logs / f"{name}.log", env)
                report = read(report_path)
            require(report["action_label"] == "command", "Wrong training target")
            require(report["status"] == ("passed" if i == 0 else "written"), "Incomplete shard")
            emit("shard_done", shard=name, steps=report["total_steps"])
            return dict(shard=name, lerobot=str(root / name / "lerobot"),
                        mcap=str(root / name / "mcap"), validation=report["validation"],
                        episodes=report["episodes"], steps=report["total_steps"])

        reports = [shard(0)]
        emit("pilot_passed", episodes=2, action_label="command", message="Starting remaining shards in parallel")
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(shard, i) for i in range(1, len(groups))]
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda r: r["shard"])
        ids = [e["episode_id"] for r in reports for e in r["episodes"]]
        require(len(ids) == len(set(ids)) == a.episodes, "Missing/duplicate episodes")
        write(root / "dataset_collection.json", dict(
            **config, status="completed", total_episodes=len(ids),
            total_steps=sum(r["steps"] for r in reports), shards=reports,
            validation="First 2 fully read back; remaining shards write-only with source integrity checks",
            training_ready=False))
        emit("campaign_done", episodes=len(ids), steps=sum(r["steps"] for r in reports))
    except Exception as exc:
        write(root / "failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
        emit("campaign_failed", error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

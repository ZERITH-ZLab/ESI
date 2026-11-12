"""Convert controller history into a small, temporally indexed evidence ledger."""
import hashlib
from pathlib import Path

from . import VERSION
from .io import digest, read, require


def discover(roots, latest_per_case=False):
    """Fixed-depth discovery avoids walking millions of observation/image files.

    Include all finished attempts, including failed attempts superseded by a retry.
    An explicitly supplied archive must also have a finalized collection receipt.
    """
    archives, excluded = set(), []
    for value in roots:
        root = Path(value).resolve()
        require(root.is_dir(), f"Input directory does not exist: {root}")
        if (root / "controller/history.json").is_file():
            archives.add(root)
            continue
        samples = list(root.glob("*/*/eval_seed_*/layout_*/replica_*/repeat_*/sample.json"))
        require(samples, f"No collection samples found under {root}; pass an experiment or attempt directory")
        for sample in sorted(samples):
            try:
                attempts = read(sample).get("attempts", [])
                # Never fall back to an older failed attempt while its retry is running.
                if latest_per_case:
                    attempts = attempts[-1:]
                for attempt in attempts:
                    raw = attempt.get("archive")
                    if not raw or attempt.get("finished_at") is None:
                        excluded.append({"sample": str(sample), "reason": "attempt has not finished"})
                        continue
                    # Resolve relative to sample for archives moved to another filesystem.
                    archive = sample.parent / Path(raw).name
                    require(archive.name.startswith("attempt_"), "Invalid attempt archive")
                    archives.add(archive.resolve())
            except (ValueError, OSError) as exc:
                excluded.append({"sample": str(sample), "reason": str(exc)})
    return sorted(archives), excluded


def prepare(archive):
    archive = Path(archive).resolve()
    sample_path = archive.parent / "sample.json"
    sample = read(sample_path)
    attempt = next((a for a in sample.get("attempts", [])
                    if Path(a.get("archive") or "").name == archive.name), None)
    require(attempt and attempt.get("finished_at") is not None, "Attempt is not finalized")
    reset = read(archive / "component_reset.json")
    require(reset.get("all_owned_processes_exited") is True and reset.get("ports_released") is True,
            "Collection process cleanup is not complete")
    blobs = {name: (archive / name).read_bytes() for name in
             ("controller/history.json", "sim/summary.json")}
    import json
    history, summary = (json.loads(blobs[name]) for name in blobs)
    # Reset records the original episode instruction, before optional online
    # subtask conditioning. Never substitute a task ID or inferred description.
    task_instruction = None
    reset_path = archive / "sim/reset.json"
    if reset_path.is_file():
        blobs["sim/reset.json"] = reset_path.read_bytes()
        initial = json.loads(blobs["sim/reset.json"])
        task_instruction = initial.get("instruction")
        require(isinstance(task_instruction, str) and bool(task_instruction.strip()),
                "Reset record has no valid original task instruction")
        require(initial.get("episode_id") == summary["episode_id"], "Reset episode identity mismatch")
    require(isinstance(history, list), "History must be a list")
    refs, decisions, events = {}, [], []
    if task_instruction is not None:
        refs["episode.task_instruction"] = dict(decision_id=-1, field="episode.task_instruction",
            value=task_instruction, available_at_step=0, phase="before")
    cursor, previous_decision = 0, -1

    def add(decision, field, value, step, phase):
        if value is None or value == "" or value == [] or value == {}:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                add(decision, f"{field}.{key}", child, step, phase)
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            for index, child in enumerate(value):
                add(decision, f"{field}.{index}", child, step, phase)
        else:
            key = f"d{decision:04d}.{field}"
            refs[key] = dict(decision_id=decision, field=field, value=value,
                             available_at_step=step, phase=phase)

    for row in history:
        d, start, end, count = (row[k] for k in ("decision", "start_tick", "end_tick", "executed_steps"))
        require(all(type(v) is int for v in (d, start, end, count)), "Noninteger timeline")
        require(d > previous_decision and start == cursor and end >= start and count == end - start,
                f"Discontinuous/invalid execution timeline at decision {d}")
        previous_decision, cursor = d, end
        response = row.get("response") or {}
        assessment = row.get("assessment") or response.get("assessment") or {}
        add(d, "assessment", assessment, start, "before")
        # Execution diagnostics only: do not expose simulator actor IDs/forces or
        # manufacture a successful grasp from the detector's internal status.
        assistance = {}
        for key in ("force_hold_steps", "tighten_steps"):
            if key in row:
                require(type(row[key]) is int and 0 <= row[key] <= count,
                        f"Invalid {key} at decision {d}")
                assistance[key] = row[key]
        holds = []
        for event in row.get("force_hold_events") or []:
            steps = event.get("steps") or []  # Post-action receipt IDs are 1-based.
            if not steps:
                continue
            require(all(type(t) is int and start < t <= end for t in steps),
                    f"Automatic hold outside executed decision {d}")
            holds.append(dict(arms=event.get("closing_arms", []),
                              executed_action_steps=[t - 1 for t in steps]))
        if holds:
            assistance["hold_intervals"] = holds
        if holds or any(assistance.values()):
            add(d, "automatic_grasp_assistance", assistance, end, "after")
        # The response schema populates inactive target/edit defaults. They are not commands.
        motion = response.get("motion")
        active = () if motion else {"eef": ("target",), "edit": ("edit",)}.get(response.get("mode"), ())
        for field in ("mode", "reason", "steps", *active):
            add(d, f"response.{field}", response.get(field), start, "before")
        if motion:
            # Motion wrappers use mode=eef but target is the INITIAL pose, not the goal.
            # Copy commanded goals only; avoid feedback/tracking fields in this snapshot.
            for field in ("tool", "arm", "waypoints"):
                add(d, f"response.motion.{field}", motion.get(field), start, "before")
        result_refs = []
        for field in ("motion_result", "correction_result"):
            result = row.get(field) or {}
            for key in ("status", "error", "arm", "executed_steps"):
                add(d, f"{field}.{key}", result.get(key), end, "after")
            result_refs += [key for key in refs if key.startswith(f"d{d:04d}.{field}.")]
        source = row.get("source", "unknown")
        control = ("pi05" if source == "student" else
                   "gpt" if source.startswith("gpt") else "unknown")
        decisions.append(dict(decision_id=d, start_step=start, end_step=end,
                              control_type=control, source=source, mode=response.get("mode")))
        # Every receipt remains auditable, even partial execution failures.
        if count == 0 or result_refs:
            events.append(dict(decision_id=d, start_step=start, end_step=end,
                               zero_step=count == 0, result_refs=result_refs))
    require(cursor == summary.get("step_id"), "History and simulator final step disagree")
    fingerprints = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    metadata = dict(source_archive=str(archive), episode_id=summary["episode_id"],
                    case=summary.get("evaluation_case"), repeat_id=sample.get("repeat_id"),
                    attempt=archive.name, collected_at=attempt["finished_at"],
                    episode_success=summary.get("success"), terminal_reason=summary.get("reason"),
                    control_dt=summary.get("control_dt"), source_sha256=fingerprints)
    payload = dict(version=VERSION, total_steps=cursor, decisions=decisions, refs=refs)
    return dict(metadata=metadata, payload=payload, events=events,
                input_hash=digest(dict(metadata=metadata, payload=payload, events=events)))


def model_payload(prepared):
    """Lossless compact presentation: timing/phase metadata once per decision.

    The full ledger remains in input.json for deterministic reference resolution.
    Episode success and collection metadata deliberately stay out of model context.
    """
    payload = prepared["payload"]
    decisions = {d["decision_id"]: dict(**d, before={}, after={}) for d in payload["decisions"]}
    for key, ref in payload["refs"].items():
        if ref["field"] == "episode.task_instruction":
            continue
        decisions[ref["decision_id"]][ref["phase"]][key] = ref["value"]
    return dict(version=payload["version"], total_steps=payload["total_steps"],
                episode_context={k: r["value"] for k, r in payload["refs"].items()
                                 if r["field"] == "episode.task_instruction"},
                decisions=list(decisions.values()))

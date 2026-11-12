"""Frozen preparation, bounded annotation/repair, resumable content-addressed runs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from queue import Queue

import jsonschema

from . import VERSION, ANNOTATION_VERSION
from .client import APIClient, CodexClient, extract_text
from .contract import SCHEMA, AnnotationValidationError, validate_and_enrich, gpt_review_report
from .io import digest, read, require, write
from .prepare import discover, prepare
from .evidence import annotation_payload
from .repair import PATCH_SCHEMA, AUDIT_SCHEMA, build_correction, merge_patch
from .audit import audit_semantics, validate_final
from .catalog import build_run_catalog, episode_name

HERE = Path(__file__).parent


@contextmanager
def batch_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another prepare/run/validate process owns this output directory") from None
        yield


def load_config(path=None):
    cfg = read(HERE / "config.example.json")
    if path:
        supplied = read(path)
        require(not (set(supplied) - set(cfg)), "Unknown config keys (credentials must not be in config)")
        cfg.update(supplied)
    require(cfg["backend"] in ("codex", "api"), "backend must be codex or api")
    require(cfg["api_mode"] in ("responses", "chat_completions"), "Invalid API mode")
    for key in ("workers", "timeout_seconds", "max_output_tokens", "max_input_chars"):
        require(type(cfg[key]) is int and cfg[key] > 0, f"Invalid {key}")
    for key in ("http_retries", "max_repairs"):
        require(type(cfg[key]) is int and 0 <= cfg[key] <= 5, f"Invalid {key}")
    require(1 <= cfg["workers"] <= 16, "workers must be 1..16")
    if cfg["backend"] == "codex":
        require(isinstance(cfg["profiles"], list) and cfg["profiles"] and
                len(set(cfg["profiles"])) == len(cfg["profiles"]), "Select distinct Codex profiles")
        require(cfg["workers"] <= len(cfg["profiles"]), "Codex workers require independently logged-in profiles")
    return cfg


def prepare_batch(roots, output, limit=None, *, append=False, latest_per_case=False,
                  prepare_fn=prepare, prepare_version=VERSION):
    output = Path(output).resolve()
    roots = [str(Path(p).resolve()) for p in roots]
    require(limit is None or limit > 0, "limit must be positive")
    require(not append or limit is None, "--append cannot be combined with --limit")
    require(not output.is_relative_to(HERE.parents[1]), "Output must be outside the code repository; use the mounted disk")
    require(all(not output.is_relative_to(Path(p)) for p in roots), "Output must be outside source archives/experiments")
    with batch_lock(output):
        selection = dict(roots=roots, limit=limit, version=prepare_version,
                         prepare_sha256=hashlib.sha256((HERE / "prepare.py").read_bytes()).hexdigest())
        if prepare_fn is not prepare:
            import inspect
            selection['route_prepare_sha256'] = hashlib.sha256(Path(inspect.getfile(prepare_fn)).read_bytes()).hexdigest()
        if latest_per_case:
            selection["latest_per_case"] = True
        manifest_path = output / "manifest.json"
        existing = None
        if manifest_path.exists():
            existing = read(manifest_path)
            require(existing["selection"] == selection, "Selection changed; use a new output directory")
            for row in existing["episodes"]:
                load_input(output, row)
            if not append:
                return existing
        archives, excluded = discover(roots, latest_per_case=latest_per_case)
        if limit:
            archives = archives[:limit]
        episodes = list(existing["episodes"]) if existing else []
        failures = []
        known = {row["archive"] for row in episodes}
        known_cases = {str(Path(path).parent): path for path in known}
        if latest_per_case:
            for archive in archives:
                previous = known_cases.get(str(archive.parent))
                require(previous is None or previous == str(archive),
                        f"A prepared case has a different latest attempt: {archive}; "
                        "append cannot replace existing inputs; use a new output directory")
        for archive in archives:
            if str(archive) in known:
                continue
            try:
                prepared = prepare_fn(archive)
                key = digest(dict(archive=str(archive), input_hash=prepared["input_hash"]))[:24]
                destination = output / "prepared" / key / "input.json"
                write(destination, prepared)
                episodes.append(dict(key=key, path=str(destination.relative_to(output)),
                                     input_hash=prepared["input_hash"], archive=str(archive),
                                     episode_id=prepared["metadata"]["episode_id"],
                                     steps=prepared["payload"]["total_steps"],
                                     decisions=len(prepared["payload"]["decisions"])))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                failures.append(dict(archive=str(archive), error=str(exc)))
        manifest = dict(selection=selection, episodes=episodes, excluded=excluded, failures=failures)
        if existing and existing != manifest:
            write(output / "manifest_history" / f"{digest(existing)[:24]}.json", existing)
        write(manifest_path, manifest)
        return manifest


def load_input(root, row, check_source=True):
    prepared = read(root / row["path"])
    body = {k: v for k, v in prepared.items() if k != "input_hash"}
    require(digest(body) == prepared["input_hash"] == row["input_hash"], "Prepared input fingerprint mismatch")
    if check_source:
        for name, checksum in prepared["metadata"]["source_sha256"].items():
            path = Path(prepared["metadata"]["source_archive"]) / name
            require(hashlib.sha256(path.read_bytes()).hexdigest() == checksum,
                    f"Source changed since preparation: {name}; prepare into a new output")
    return prepared


def protocol():
    # Code changes invalidate caches as well as prompt/schema changes.
    sources = {p.name: p.read_text() for p in sorted(HERE.glob("*.py"))}
    return dict(version=ANNOTATION_VERSION, prompt=(HERE / "prompt.md").read_text(), schema=SCHEMA,
                repair_schema=PATCH_SCHEMA, audit_schema=AUDIT_SCHEMA,
                implementation_sha256=digest(sources))


def repair_changes(previous, current):
    """Audit what a repair changed; do not silently patch labels on the model's behalf."""
    def rows(value):
        if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
            return {}
        return {s["segment_id"]: s for s in value["segments"]
                if isinstance(s, dict) and isinstance(s.get("segment_id"), str)}
    before, after = rows(previous), rows(current)
    changes = []
    for sid in sorted(before.keys() | after.keys()):
        if sid not in before or sid not in after:
            changes.append(dict(segment_id=sid, change="added" if sid in after else "removed"))
        elif before[sid] != after[sid]:
            fields = [k for k in sorted(before[sid].keys() | after[sid].keys())
                      if before[sid].get(k) != after[sid].get(k)]
            changes.append(dict(segment_id=sid, change="modified", fields=fields,
                                boundaries_changed=any(k in fields for k in ("start_step", "end_step"))))
    return changes


def annotate_one(prepared, destination, cfg, spec, client):
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / "annotation.json"
    if completed.exists():
        cached = read(completed)
        require(cached["input_hash"] == prepared["input_hash"], "Cached result belongs to another input")
        candidate = read(destination / "candidate.json")
        require(validate_final(prepared, candidate, read(destination / "audit_final.json")) == cached,
                "Cached annotation was modified or corrupted")
        return cached, "reused"
    payload = annotation_payload(prepared)
    write(destination / "endpoint_feedback_contexts.json", payload["endpoint_feedback_contexts"])
    require(len(json.dumps(payload, ensure_ascii=False)) <= cfg["max_input_chars"],
            "Episode exceeds configured input size; no silent truncation/windowing")
    correction = None
    for attempt in range(cfg["max_repairs"] + 1):
        raw_path = destination / f"response_{attempt:02d}.json"
        if raw_path.exists():
            raw = read(raw_path)
        else:
            raw = client.complete(spec["prompt"], payload, correction)
            write(raw_path, raw)
        candidate = correction["previous"] if correction is not None else None
        try:
            output = extract_text(raw, "codex" if cfg["backend"] == "codex" else cfg["api_mode"])
            if correction is not None and correction["mode"] == "patch":
                write(destination / f"patch_{attempt:02d}.json", output)
                candidate = merge_patch(correction["previous"], output, correction["editable_segment_ids"])
            else:
                candidate = output
            # Human-readable full draft is saved even when validation rejects the episode.
            write(destination / f"draft_{attempt:02d}.json", candidate)
            if correction is not None:
                write(destination / f"repair_changes_{attempt:02d}.json",
                      repair_changes(correction["previous"], candidate))
            annotation = validate_and_enrich(prepared, candidate)
            write(destination / f"validation_{attempt:02d}.json", dict(valid=True, errors=[]))
            break
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
            error = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
            errors = (exc.errors if isinstance(exc, AnnotationValidationError) else
                      [dict(code="invalid_response", segment_id=None, field="response", message=error)])
            write(destination / f"validation_{attempt:02d}.json", dict(valid=False, error=error, errors=errors))
            if attempt == cfg["max_repairs"]:
                # Only validly parsed/merged label drafts can receive the pi05
                # exception. A malformed response/patch is still a hard failure.
                if isinstance(exc, AnnotationValidationError) and not any(
                        e["code"] == "invalid_patch" for e in errors):
                    try:
                        annotation = validate_and_enrich(prepared, candidate, allow_pi05_exclusions=True)
                    except AnnotationValidationError as final_exc:
                        errors = final_exc.errors
                        error = str(final_exc)
                        write(destination / f"validation_{attempt:02d}.json",
                              dict(valid=False, error=error, errors=exc.errors, blocking_errors=errors))
                    else:
                        write(destination / f"validation_{attempt:02d}.json", dict(
                            valid=False, accepted_with_exclusions=True, errors=errors,
                            excluded_segment_ids=annotation["excluded_segment_ids"]))
                        break
                raise ValueError(f"Annotation failed validation after {attempt + 1} attempts: {error}") from None
            correction = build_correction(prepared, candidate if candidate is not None else raw, errors)
            write(destination / f"repair_request_{attempt + 1:02d}.json", correction)
    else:
        raise AssertionError("unreachable")
    # Persist the accepted base separately. The candidate/pool are written only
    # after the mandatory semantic audit, sharing the remaining repair budget.
    write(destination / "pre_audit_candidate.json", candidate)
    candidate, annotation = audit_semantics(prepared, candidate, annotation, destination,
                                            cfg, spec, client, cfg["max_repairs"] - attempt)
    write(destination / "candidate.json", candidate)
    write(completed, annotation)
    return annotation, "annotated_with_exclusions" if annotation["excluded_segment_ids"] else "annotated"


def run_batch(root, cfg, limit=None):
    root = Path(root).resolve()
    require(limit is None or limit > 0, "limit must be positive")
    with batch_lock(root), ExitStack() as stack:
        manifest = read(root / "manifest.json")
        rows = manifest["episodes"][:limit]
        require(rows, "No prepared episodes")
        spec = protocol()
        # Runtime concurrency and timeout do not change the annotation contract.
        identity = {k: v for k, v in cfg.items() if k not in ("workers", "timeout_seconds", "http_retries", "profiles", "proxy_file", "api_key_env")}
        run_id = digest(dict(protocol=spec, config=identity))[:20]
        run_root = root / "runs" / run_id
        write(run_root / "protocol.json", spec)
        write(run_root / "config.json", cfg)
        slots = Queue()
        if cfg["backend"] == "codex":
            from hybrid_rollout.robodojo.codex_backend.profiles import account_lock, validate_campaign_sessions
            profiles = cfg["profiles"][:cfg["workers"]]
            validate_campaign_sessions(cfg["shared_root"], tuple(profiles))
            for profile in profiles:
                lease = stack.enter_context(account_lock(profile, cfg["shared_root"]))
                slots.put((profile, lease))
        else:
            for _ in range(cfg["workers"]):
                slots.put(None)

        def worker(row):
            slot = slots.get()
            try:
                prepared = load_input(root, row)
                destination = run_root / "episodes" / row["key"]
                client = (CodexClient(cfg, *slot, destination / "codex_runtime")
                          if slot else APIClient(cfg))
                annotation, state = annotate_one(prepared, destination, cfg, spec, client)
                attention = gpt_review_report(annotation)
                write(destination / "gpt_review.json", attention)
                if attention["flagged_segments"]:
                    print(json.dumps(dict(event="gpt_annotation_attention", severity="warning", key=row["key"],
                                          report=str(destination / "gpt_review.json"), **attention),
                                     ensure_ascii=False), flush=True)
                result = dict(key=row["key"], state=state, segments=len(annotation["segments"]),
                              custom=sum((s.get("instruction") or {}).get("skill_type") == "custom"
                                         for s in annotation["segments"]),
                              **attention["counts"],
                              excluded_pi05=len(annotation["excluded_segment_ids"]),
                              unknown=sum(s["outcome"] == "unknown" for s in annotation["segments"]),
                              inferred=sum(s["outcome_basis"] == "inferred" for s in annotation["segments"]),
                              semantic_reviewed=annotation["semantic_audit"]["reviewed_segments"],
                              needs_review=sum(s["needs_review"] for s in annotation["segments"]),
                              training_eligible=sum(s["training_eligible"] for s in annotation["segments"]),
                              high_level_eligible=sum(s["training_eligibility"]["high_level"] for s in annotation["segments"]))
            except Exception as exc:
                result = dict(key=row["key"], state="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                slots.put(slot)
            result.update(name=episode_name(row), source_archive=row["archive"],
                          annotation=str(run_root / "episodes" / row["key"] / "annotation.json")
                          if result["state"] != "failed" else None)
            write(run_root / "episodes" / row["key"] / "status.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            results = list(pool.map(worker, rows))
        summary = dict(run_id=run_id, run_root=str(run_root), results=results,
                       failures=sum(r["state"] == "failed" for r in results))
        write(run_root / "summary.json", summary)
        # Rebuild index from ALL valid cached episodes, including earlier --limit runs.
        index = []
        for row in manifest["episodes"]:
            annotation_path = run_root / "episodes" / row["key"] / "annotation.json"
            if not annotation_path.exists():
                continue
            try:
                prepared = load_input(root, row)
                annotation = read(annotation_path)
                require(validate_final(prepared, read(annotation_path.with_name("candidate.json")),
                                       read(annotation_path.with_name("audit_final.json"))) == annotation,
                        "Invalid cached annotation")
                for s in annotation["segments"]:
                    if s["annotation_status"] == "excluded_pi05":
                        continue
                    index.append(dict(episode_id=row["episode_id"], source_archive=row["archive"],
                                      annotation=str(annotation_path), episode_success=annotation["metadata"]["episode_success"],
                                      collected_at=annotation["metadata"]["collected_at"], **s))
            except (ValueError, OSError, KeyError, jsonschema.ValidationError):
                continue  # Failed/stale artifacts never enter the sampling index.
        write(run_root / "segments.json", index)
        catalog = build_run_catalog(root, run_root, manifest)
        summary.update(named_run_root=catalog["named_run_root"],
                       episode_index=str(run_root / "episodes.index.csv"))
        write(run_root / "summary.json", summary)
        return summary


def validate_run(root, run_id):
    root = Path(root).resolve()
    require(Path(run_id).name == run_id, "Invalid run ID")
    with batch_lock(root):
        manifest = read(root / "manifest.json")
        run_root = root / "runs" / run_id
        require(run_root.is_dir(), "Run does not exist")
        require(read(run_root / "protocol.json")["version"] == ANNOTATION_VERSION,
                "Run uses an older annotation contract; keep its drafts for review and run the new contract separately")
        reports = []
        for row in manifest["episodes"]:
            destination = run_root / "episodes" / row["key"]
            try:
                prepared = load_input(root, row)
                expected = validate_final(prepared, read(destination / "candidate.json"),
                                          read(destination / "audit_final.json"))
                require(expected == read(destination / "annotation.json"), "Annotation differs from validated candidate")
                reports.append(dict(key=row["key"], state="valid", excluded_pi05=len(expected["excluded_segment_ids"])))
            except (ValueError, OSError, KeyError, jsonschema.ValidationError) as exc:
                reports.append(dict(key=row["key"], state="failed", error=str(exc)))
        report = dict(results=reports, failures=sum(r["state"] == "failed" for r in reports))
        write(run_root / "validation.json", report)
        return report

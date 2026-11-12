"""Export extraction v2 to local LeRobot v3 (0.4.4) and JSON-schema MCAP.

No upload or training. Requires an explicit training action-label choice.
Original RGB is lossless in LeRobot; MCAP uses H264 video streams.
Normalized grippers use an explicit custom topic, NOT fake rad/m JointStates.
"""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np

from .extract_episode import CAMERAS, VERSION, read, require, write

EXPORT_VERSION = "rollout_dual_export.v1"
CAMERA_NAMES = dict(zip(CAMERAS, ("cam_high_0", "cam_wrist_0", "cam_wrist_1")))
NAMES = [
    f"{arm}_joint{i}" if i < 7 else f"{arm}_gripper_normalized"
    for arm in ("left", "right")
    for i in range(1, 8)
]
H264 = dict(
    preset="medium",
    crf="23",
    bf="0",
    g="30",
    keyint_min="30",
    sc_threshold="0",
    annexb="1",
    **{"x264-params": "repeat-headers=1"},
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(index):
    data = read(index)
    require(
        data["failed"] == 0 and data["results"],
        "Input batch contains failed/no exports",
    )
    result, seen = [], set()
    for row in data["results"]:
        p = Path(row["output"]).resolve()
        m = read(p / "manifest.json")
        require(m["schema"] == VERSION, "Expected extraction v2")
        require(m["episode_id"] not in seen, "Duplicate episode")
        seen.add(m["episode_id"])
        for name, sha in m["file_sha256"].items():
            require(
                digest(p / name) == sha, f"Extraction checksum mismatch: {p / name}"
            )
        require(
            read(p / "report.json")["status"] == "passed", "Extraction not validated"
        )
        result.append(p)
    return result


def image_at(archive, ref):
    path = (archive / ref["path"]).resolve()
    require(path.is_relative_to(archive), "Unsafe observation reference")
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == ref["sha256"], f"Observation changed: {path}"
    )
    with np.load(io.BytesIO(raw), allow_pickle=False) as obs:
        return {k: obs[k].copy() for k in CAMERAS}


def prompt(episode, decision, mode):
    text = episode["task_instruction"]
    if mode == "subtask":
        a = decision["response"].get("assessment", {})
        text += "\nCurrent subtask: " + a.get("current_subgoal", "")
        # Only contemporaneous evidence, never episode-end summaries/outcomes.
        if a.get("execution_status") == "failed":
            text += "\nPrevious execution issue: " + a.get("execution_evidence", "")
    return text


def ns(t, dt):
    return round(t * dt * 1_000_000_000)


def timestamp(t):
    return dict(sec=t // 1_000_000_000, nsec=t % 1_000_000_000)


def obj(properties):
    return dict(type="object", properties=properties, required=list(properties))


def schemas():
    number, string = dict(type="number"), dict(type="string")
    stamp = obj(
        dict(
            sec=dict(type="integer", minimum=0),
            nsec=dict(type="integer", minimum=0, maximum=999999999),
        )
    )
    array = lambda items: dict(type="array", items=items)
    return {
        "foxglove.JointStates": obj(
            dict(timestamp=stamp, joints=array(obj(dict(name=string, position=number))))
        ),
        "foxglove.PoseInFrame": obj(
            dict(
                timestamp=stamp,
                frame_id=string,
                pose=obj(
                    dict(
                        position=obj(dict.fromkeys("xyz", number)),
                        orientation=obj(dict.fromkeys("xyzw", number)),
                    )
                ),
            )
        ),
        "foxglove.CompressedVideo": obj(
            dict(
                timestamp=stamp,
                frame_id=string,
                data=dict(type="string", contentEncoding="base64"),
                format=string,
            )
        ),
        "custom.TextMap": obj(
            dict(timestamp=stamp, fields=array(obj(dict(key=string, value=string))))
        ),
        "custom.NormalizedGripper": obj(dict(timestamp=stamp, opening=number)),
    }


class McapEpisode:
    def __init__(self, path, dt, shapes):
        import av
        from mcap.writer import Writer

        self.stream = path.open("xb")
        self.writer = Writer(self.stream)
        self.writer.start(profile="json")
        self.ids = {
            name: self.writer.register_schema(
                name, "jsonschema", json.dumps(dict(title=name, **schema)).encode()
            )
            for name, schema in schemas().items()
        }
        self.channels, self.counts = {}, Counter()
        self.dt, self.codecs = dt, {}
        for k, shape in shapes.items():
            c = av.CodecContext.create("libx264", "w")
            c.height, c.width = shape[:2]
            c.pix_fmt = "yuv420p"
            c.time_base = Fraction(str(dt))
            c.framerate = 1 / c.time_base
            c.thread_count = 2
            c.options = H264.copy()
            c.open()
            self.codecs[k] = c

    def send(self, topic, schema, t, value):
        if topic not in self.channels:
            self.channels[topic] = self.writer.register_channel(
                topic, "json", self.ids[schema]
            )
        self.writer.add_message(
            self.channels[topic],
            t,
            json.dumps(dict(timestamp=timestamp(t), **value), allow_nan=False).encode(),
            t,
        )
        self.counts[topic] += 1

    def packets(self, key, packets):
        for packet in packets:
            require(packet.pts is not None, "Missing H264 PTS")
            raw = bytes(packet)
            require(raw.startswith((b"\0\0\0\1", b"\0\0\1")), "H264 is not Annex B")
            self.send(
                f"observation/images/{CAMERA_NAMES[key]}/color_h264",
                "foxglove.CompressedVideo",
                ns(packet.pts, self.dt),
                dict(
                    frame_id=CAMERA_NAMES[key],
                    format="h264",
                    data=base64.b64encode(raw).decode(),
                ),
            )

    def frame(self, t, images, arrays, decision, episode):
        import av

        stamp = ns(t, self.dt)
        n = len(arrays["command"])
        for side, offset in enumerate((0, 7)):
            for prefix, values in [
                ("observation/state", arrays["observation_state"][t])
            ] + ([("action", arrays["command"][t])] if t < n else []):
                self.send(
                    f"{prefix}/arm_{side}/joint_state",
                    "foxglove.JointStates",
                    stamp,
                    dict(
                        joints=[
                            dict(
                                name=f"joint{i + 1}", position=float(values[offset + i])
                            )
                            for i in range(6)
                        ]
                    ),
                )
                self.send(
                    f"{prefix}/effector_{side}/normalized_opening",
                    "custom.NormalizedGripper",
                    stamp,
                    dict(opening=float(values[offset + 6])),
                )
            p, q = (
                arrays["observation_eef_positions"][t, side],
                arrays["observation_eef_quaternions_wxyz"][t, side],
            )
            self.send(
                f"observation/state/end_{side}/pose",
                "foxglove.PoseInFrame",
                stamp,
                dict(
                    frame_id="environment_origin",
                    pose=dict(
                        position=dict(zip("xyz", map(float, p))),
                        orientation=dict(zip("xyzw", map(float, q[[1, 2, 3, 0]]))),
                    ),
                ),
            )
        if t == 0 or (
            t < n and (decision["decision_id"] != int(arrays["decision_id"][t - 1]))
        ):
            assessment = decision["response"].get("assessment", {})
            fields = dict(
                instruction=episode["task_instruction"],
                language="en",
                subtask=assessment.get("current_subgoal", ""),
                reason=decision["response"].get("reason", ""),
                execution_evidence=assessment.get("execution_evidence", ""),
                mode=decision["mode"],
                motion_kind=decision["motion_kind"],
                decision_id=str(decision["decision_id"]),
            )
            self.send(
                "observation/instruction",
                "custom.TextMap",
                stamp,
                dict(fields=[dict(key=k, value=v) for k, v in fields.items()]),
            )
        for k, image in images.items():
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = t, Fraction(str(self.dt))
            self.packets(k, self.codecs[k].encode(frame))

    def finish(self):
        for key, codec in self.codecs.items():
            self.packets(key, codec.encode(None))
        self.writer.finish()
        self.stream.close()


def verify_mcap(path, arrays, dt):
    import av
    from mcap.reader import make_reader

    counts, previous, decoders, decoded = Counter(), {}, {}, Counter()
    n = len(arrays["command"])
    with path.open("rb") as stream:
        for schema, channel, message in make_reader(
            stream, validate_crcs=True
        ).iter_messages():
            v = json.loads(message.data)
            topic = channel.topic
            t = round(message.log_time / (dt * 1e9))
            require(
                v["timestamp"] == timestamp(message.log_time), "MCAP timestamp mismatch"
            )
            require(message.log_time >= previous.get(topic, -1), "Nonmonotonic topic")
            previous[topic] = message.log_time
            require(abs(message.log_time - ns(t, dt)) <= 1, "Off-grid timestamp")
            if schema.name == "foxglove.CompressedVideo":
                codec = decoders.setdefault(topic, av.CodecContext.create("h264", "r"))
                frames = codec.decode(av.Packet(base64.b64decode(v["data"])))
                decoded[topic] += len(frames)
            elif schema.name == "foxglove.JointStates":
                side = 0 if "/arm_0/" in topic else 1
                ref = (
                    arrays["command"][t]
                    if topic.startswith("action/")
                    else arrays["observation_state"][t]
                )
                np.testing.assert_allclose(
                    [j["position"] for j in v["joints"]],
                    ref[side * 7 : side * 7 + 6],
                    atol=1e-7,
                )
            elif schema.name == "custom.NormalizedGripper":
                side = 0 if "/effector_0/" in topic else 1
                ref = (
                    arrays["command"][t]
                    if topic.startswith("action/")
                    else arrays["observation_state"][t]
                )
                np.testing.assert_allclose(v["opening"], ref[side * 7 + 6], atol=1e-7)
            elif schema.name == "foxglove.PoseInFrame":
                side = 0 if "/end_0/" in topic else 1
                np.testing.assert_allclose(
                    list(v["pose"]["position"].values()),
                    arrays["observation_eef_positions"][t, side],
                    atol=1e-7,
                )
                q = v["pose"]["orientation"]
                np.testing.assert_allclose(
                    [q[k] for k in "wxyz"],
                    arrays["observation_eef_quaternions_wxyz"][t, side],
                    atol=1e-7,
                )
            counts[topic] += 1
    for topic, codec in decoders.items():
        decoded[topic] += len(codec.decode(None))
        require(decoded[topic] == n + 1, "Decoded MCAP frame count mismatch")
    require(len(decoded) == 3, "Missing camera")
    expected_topics = {"observation/instruction"}
    expected_topics.update(
        f"observation/images/{v}/color_h264" for v in CAMERA_NAMES.values()
    )
    for side in (0, 1):
        expected_topics.add(f"observation/state/end_{side}/pose")
        for prefix in ("action", "observation/state"):
            expected_topics.update(
                (
                    f"{prefix}/arm_{side}/joint_state",
                    f"{prefix}/effector_{side}/normalized_opening",
                )
            )
    require(set(counts) == expected_topics, "Missing/unexpected MCAP topics")
    for topic, count in counts.items():
        if topic != "observation/instruction":
            require(
                count == (n if topic.startswith("action/") else n + 1),
                f"Message count: {topic}",
            )
    return dict(status="passed", messages=dict(counts), decoded_frames=dict(decoded))


def export(index, output, action_label, text_mode="task", formats="both", validation="full"):
    require(action_label in ("command", "next_state"), "Select an action label")
    require(validation in ("full", "write_only"), "Invalid validation mode")
    output = Path(output).resolve()
    inputs = sources(index)
    require(
        not any(output.is_relative_to(p) for p in inputs),
        "Do not write inside extraction",
    )
    output.mkdir(parents=True, exist_ok=False)
    dataset, summaries = None, []
    dt = read(inputs[0] / "manifest.json")["control_dt"]
    fps = round(1 / dt)
    require(abs(fps * dt - 1) < 1e-8, "LeRobot requires an integer fixed fps")
    shapes = read(inputs[0] / "report.json")["image_shapes"]
    export_info = dict(
        version=EXPORT_VERSION,
        input_index=str(Path(index).resolve()),
        input_sha256=digest(Path(index)),
        action_label=action_label,
        text_mode=text_mode,
        fps=fps,
        formats=formats,
        validation=validation,
        dependencies={
            k: importlib.metadata.version(k) for k in ("numpy", "av", "lerobot", "mcap")
        },
    )
    write(output / "export_config.json", export_info)
    if formats in ("both", "lerobot"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            k: dict(dtype="float32", shape=(14,), names=NAMES)
            for k in (
                "observation.state",
                "action",
                "recorded.command",
                "recorded.next_state",
            )
        }
        features.update(
            {
                f"observation.images.{k}": dict(
                    dtype="image",
                    shape=(3, *shape[:2]),
                    names=["channels", "height", "width"],
                )
                for k, shape in shapes.items()
            }
        )
        features["decision_id"] = dict(dtype="int64", shape=(1,), names=["decision_id"])
        dataset = LeRobotDataset.create(
            repo_id="local/robodojo_rollout",
            root=output / "lerobot",
            robot_type="dual_arx_x5",
            fps=fps,
            features=features,
            use_videos=False,
            image_writer_threads=4,
            image_writer_processes=0,
        )
    try:
        for i, p in enumerate(inputs):
            manifest, episode = read(p / "manifest.json"), read(p / "episode.json")
            require(manifest["control_dt"] == dt, "Mixed control frequencies")
            require(
                read(p / "report.json")["image_shapes"] == shapes, "Mixed image shapes"
            )
            with np.load(p / "steps.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            decisions = {d["decision_id"]: d for d in read(p / "decisions.json")}
            refs = read(p / "observations.json")
            n = manifest["steps"]
            archive = Path(manifest["source_archive"]).resolve()
            sidecar = output / "research" / f"episode_{i:06d}"
            sidecar.mkdir(parents=True)
            for name in manifest["files"] + ["manifest.json"]:
                shutil.copyfile(p / name, sidecar / name)
            task = episode["summary"]["evaluation_case"]["task"]
            mcap, mpath = None, None
            if formats in ("both", "mcap"):
                taskroot = output / "mcap" / task
                taskroot.mkdir(parents=True, exist_ok=True)
                if not (taskroot / "task_metadata.json").exists():
                    write(
                        taskroot / "task_metadata.json",
                        dict(
                            project_id=None,
                            project_name="GPT-as-Policy research",
                            task_id=task,
                            task_name=task,
                            action_desc=episode["task_instruction"],
                            ingest_ready=False,
                        ),
                    )
                uid = (
                    datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                    + "_simulation_"
                    + manifest["episode_id"][:8]
                )
                folder = taskroot / uid
                folder.mkdir()
                mpath = folder / f"{uid}.mcap"
                mcap = McapEpisode(mpath, dt, shapes)
                write(
                    folder / "metadata.json",
                    dict(
                        collect_data_id=uid,
                        task_id=task,
                        robot_type="dual_arx_x5",
                        version=EXPORT_VERSION,
                        policy_version=episode["provenance"].get("resume_identity"),
                        outcome="success"
                        if episode["summary"]["success"]
                        else "failure",
                        data_type="simulation",
                        instruction=episode["task_instruction"],
                        episode_id=manifest["episode_id"],
                        time_basis="episode-relative simulation nanoseconds; filename time is export time",
                        dt=dt,
                        gripper_units="normalized; exported on custom normalized_opening topics",
                        pose_frame="environment_origin",
                        pose_reference="link6",
                        h264_options=H264,
                        ingest_ready=False,
                        missing_for_company_ingest=[
                            "registered project/task IDs",
                            "camera calibration and TF",
                            "verified gripper rad/m mapping",
                            "base_link0/1 pose mapping",
                        ],
                        source_extract=str(p),
                        research_sidecar=str(sidecar.relative_to(output)),
                    ),
                )
            try:
                for t, ref in enumerate(refs):
                    images = image_at(archive, ref)
                    d = decisions[int(arrays["decision_id"][min(t, n - 1)])]
                    if dataset is not None and t < n:
                        dataset.add_frame(
                            dict(
                                {
                                    "observation.state": arrays["state"][t],
                                    "action": arrays[action_label][t],
                                    "recorded.command": arrays["command"][t],
                                    "recorded.next_state": arrays["next_state"][t],
                                    "decision_id": np.array(
                                        [d["decision_id"]], dtype=np.int64
                                    ),
                                    "task": prompt(episode, d, text_mode),
                                },
                                **{
                                    f"observation.images.{k}": v
                                    for k, v in images.items()
                                },
                            )
                        )
                    if mcap:
                        mcap.frame(t, images, arrays, d, episode)
                    if t % 200 == 0:
                        print(
                            json.dumps(
                                dict(
                                    event="export_progress", episode=i, step=t, total=n
                                )
                            ),
                            flush=True,
                        )
                if mcap:
                    mcap.finish()
            finally:
                if mcap and not mcap.stream.closed:
                    mcap.stream.close()
            if dataset is not None:
                dataset.save_episode(parallel_encoding=False)
            checked = (verify_mcap(mpath, arrays, dt) if validation == "full"
                       else dict(status="not_run", reason="Full readback disabled after pilot validation")) if mpath else None
            summaries.append(
                dict(
                    episode_index=i,
                    episode_id=manifest["episode_id"],
                    steps=n,
                    mcap=str(mpath.relative_to(output)) if mpath else None,
                    mcap_validation=checked,
                )
            )
            write(sidecar / "export_receipt.json", summaries[-1])
    finally:
        if dataset is not None:
            dataset.finalize()
    if dataset is not None and validation == "full":
        lerobot_check = verify_lerobot(output / "lerobot", inputs, action_label, dt)
    elif dataset is not None:
        lerobot_check = dict(status="not_run", reason="Full readback disabled after pilot validation")
    else:
        lerobot_check = None
    result = dict(
        **export_info,
        status="passed" if validation == "full" else "written",
        episodes=summaries,
        total_steps=sum(s["steps"] for s in summaries),
        lerobot_validation=lerobot_check,
        training_ready=False,
        warning="Unfiltered dataset; command denotes the executed target, not measured next state. Training transforms, normalization and padding policy still require review.",
    )
    write(output / "export_manifest.json", result)
    return result


def verify_lerobot(root, inputs, action_label, dt):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    columns = [
        "action",
        "observation.state",
        "recorded.command",
        "recorded.next_state",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
    ]
    table = pa.concat_tables(
        [
            pq.read_table(p, columns=columns)
            for p in sorted((root / "data").rglob("*.parquet"))
        ]
    ).sort_by("index")
    numeric = table.to_pydict()
    ds = LeRobotDataset(
        "local/robodojo_rollout",
        root=root,
        delta_timestamps={"action": [j * dt for j in range(50)]},
    )
    offset, checked = 0, []
    for episode_index, p in enumerate(inputs):
        with np.load(p / "steps.npz") as a:
            n = len(a["command"])
            span = slice(offset, offset + n)
            for feature, key in [
                ("action", action_label),
                ("observation.state", "state"),
                ("recorded.command", "command"),
                ("recorded.next_state", "next_state"),
            ]:
                np.testing.assert_allclose(numeric[feature][span], a[key], atol=1e-6)
            np.testing.assert_array_equal(
                numeric["episode_index"][span], np.full(n, episode_index)
            )
            np.testing.assert_array_equal(numeric["frame_index"][span], np.arange(n))
            np.testing.assert_allclose(
                numeric["timestamp"][span], np.arange(n) * dt, atol=1e-5
            )
            for t in sorted({0, n // 2, n - 1}):
                sample = ds[offset + t]
                expected = a[action_label][np.minimum(np.arange(t, t + 50), n - 1)]
                np.testing.assert_allclose(
                    sample["action"].numpy(), expected, atol=1e-6
                )
                np.testing.assert_allclose(
                    sample["observation.state"].numpy(), a["state"][t], atol=1e-6
                )
                require(tuple(sample["action"].shape) == (50, 14), "Wrong action chunk")
                refs = read(p / "observations.json")
                m = read(p / "manifest.json")
                images = image_at(Path(m["source_archive"]), refs[t])
                for k in CAMERAS:
                    recovered = np.rint(
                        sample[f"observation.images.{k}"].numpy().transpose(1, 2, 0)
                        * 255
                    ).astype(np.uint8)
                    require(np.array_equal(recovered, images[k]), "LeRobot RGB differs")
                if t == n - 1:
                    require(
                        int(sample["action_is_pad"].sum()) == 49,
                        "Wrong tail padding mask",
                    )
            offset += n
            checked.append(n)
    require(len(ds) == offset, "Wrong training row count")
    return dict(
        status="passed",
        episodes=len(inputs),
        rows=offset,
        chunk_shape=[50, 14],
        checks="all numeric rows; start/middle/end chunks and lossless RGB; terminal padding; no cross-episode chunks",
        tail_policy="LeRobot repeats last action and returns action_is_pad; training mask handling must be reviewed",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--action-label", required=True, choices=["command", "next_state"])
    p.add_argument("--text-mode", default="task", choices=["task", "subtask"])
    p.add_argument("--formats", default="both", choices=["both", "lerobot", "mcap"])
    p.add_argument("--validation", default="full", choices=["full", "write_only"])
    a = p.parse_args()
    result = export(a.index, a.output, a.action_label, a.text_mode, a.formats, a.validation)
    print(
        json.dumps(
            dict(
                status=result["status"],
                episodes=len(result["episodes"]),
                steps=result["total_steps"],
            )
        )
    )


if __name__ == "__main__":
    main()

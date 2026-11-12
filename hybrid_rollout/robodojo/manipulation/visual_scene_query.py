"""SceneQuery with in-memory RGB, instance masks and scene-object identity.

Only query() is public. Call on the simulation thread between physics steps.
Images are NumPy arrays, not JSON lists or paths. No files or robot commands
are produced. Rendering is allowed; advancing physics is not.
"""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any

import numpy as np

from .scene_query import SceneQuery


def _map_segments(instance_ids, id_to_paths, records):
    """Join visible leaf prims to exact registered ancestors."""
    if instance_ids.ndim != 2 or instance_ids.dtype != np.uint32:
        raise ValueError("instance_ids must be an H x W uint32 array")
    if not isinstance(id_to_paths, Mapping):
        raise TypeError("Missing native idToLabels path mapping")
    roots = {}
    for _, record in records:
        root = record["prim_path"]
        if not isinstance(root, str) or not root.startswith("/"):
            raise ValueError("Registered object needs an absolute prim_path")
        root = root.rstrip("/")
        if not root or root in roots:
            raise ValueError("Invalid or duplicate object prim_path")
        roots[root] = record["inst_name"]
    paths = {int(key): value for key, value in id_to_paths.items()}
    mapped, unmapped, groups, visible_paths = {}, {}, {}, {}
    for segment in np.unique(instance_ids):
        segment = int(segment)
        if segment not in paths:
            raise ValueError(f"No native prim path for segment {segment}")
        path = paths[segment]
        if not isinstance(path, str):
            raise TypeError("Native segmentation labels must be prim paths")
        visible_paths[segment] = path
        matches = [
            root
            for root in roots
            if path == root or path.startswith(root + "/")
        ]
        if not matches:
            unmapped[segment] = path
            continue
        # A leaf belongs to its deepest registered ancestor, not a similarly
        # named sibling (e.g. block_1 must not capture block_10).
        object_id = roots[max(matches, key=len)]
        mapped[segment] = object_id
        groups.setdefault(object_id, []).append(segment)
    visible = []
    for object_id, segments in sorted(groups.items()):
        ys, xs = np.nonzero(np.isin(instance_ids, segments))
        visible.append(
            {
                "object_id": object_id,
                "segment_ids": segments,
                "pixel_count": int(len(xs)),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
            }
        )
    return {
        "segment_to_object": mapped,
        "segment_paths": visible_paths,
        "unmapped_segments": unmapped,
        "visible_objects": visible,
    }


class VisualSceneQuery(SceneQuery):
    """Extend SceneQuery without changing the original class or its fields.

    Args:
        env: Initialized and reset native RoboDojo environment.

    Enable native zero-delay rendering settings before AppLauncher; query()
    fails explicitly if they are absent. The caller must serialize rendering,
    reset and physics updates on the simulation thread.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self._env = env

    def query(
        self,
        identifiers: str | Sequence[str] | None = None,
        *,
        fields: Sequence[str] = ("pose", "description"),
        env_idx: int = 0,
        camera: str,
    ) -> dict:
        """Return original object fields plus one camera's visual snapshot.

        Args:
            identifiers: Original exact label/ID selector; None selects all.
            fields: Original SceneQuery fields to include.
            env_idx: Native sub-environment index.
            camera: Exact camera name, e.g. cam_head; no guessed aliases.

        Returns:
            Original envelope plus visual: rgb (H,W,3 uint8), instance_ids
            (H,W uint32), segment_to_object (int -> original object ID),
            segment_paths, unmapped_segments, visible_objects and capture
            metadata. Arrays own their memory and survive later queries.
            Bounding boxes use image x/y pixels, exclusive right/bottom edges.
            Images/mappings cover the full camera view, even when identifiers
            filters the returned object state. Only registered task objects
            can be mapped; absence from visible_objects is not proof of absence
            from the scene. Renderer segment numbers are snapshot-local.

        Raises:
            KeyError: Unknown camera, or an original selector error.
            ValueError: Invalid arguments, state, image or mapping.
            RuntimeError: Missing render synchronization or physics advanced.

        This returns privileged simulation ground truth, not learned visual
        recognition. It does not interpret phrases such as 'left blue bottle'.
        """
        before = int(self._env.sim.sim.current_time_step_index)
        packet = super().query(identifiers, fields=fields, env_idx=env_idx)
        if not isinstance(camera, str) or not camera:
            raise ValueError("camera must be a non-empty native camera name")
        names = self._env.camera_manager.camera_names[env_idx]
        matches = [idx for idx, name in enumerate(names) if name == camera]
        if len(matches) != 1:
            raise KeyError(
                f"Camera {camera!r} is not unique; available={names}"
            )
        native_camera = self._env.camera_manager.cameras[env_idx][matches[0]]
        try:
            rgb, instances, paths, capture = self._capture(native_camera)
            mapping = _map_segments(instances, paths, self._records(env_idx))
        finally:
            if int(self._env.sim.sim.current_time_step_index) != before:
                raise RuntimeError("Visual query unexpectedly advanced physics")
        packet["visual"] = {
            "camera": camera,
            "env_idx": int(env_idx),
            "physics_step": before,
            "pixel_frame": "image_xy_top_left",
            "rgb": rgb,
            "instance_ids": instances,
            **mapping,
            **capture,
        }
        return packet

    def _capture(self, camera):
        """Render RGB and leaf-instance IDs on one isolated render product."""
        import omni.replicator.core as rep
        from env.camera_manager.capture.render_sync import (
            dispatcher_time,
            zero_delay_active,
            zero_delay_kit_args,
        )

        if not zero_delay_active():
            raise RuntimeError(
                "Launch AppLauncher with zero-delay --kit_args: "
                + zero_delay_kit_args()
            )
        width, height = camera.get_resolution()
        with ExitStack() as resources:
            # force_new isolates attach/detach from the existing RGB pipeline.
            product = rep.create.render_product(
                camera.prim_path, (width, height), force_new=True
            )
            resources.callback(product.destroy)
            rgb_reader = rep.AnnotatorRegistry.get_annotator(
                "rgb", device="cpu", do_array_copy=True
            )
            instance_reader = rep.AnnotatorRegistry.get_annotator(
                "instance_id_segmentation_fast",
                init_params={"colorize": False},
                device="cpu",
                do_array_copy=True,
            )
            for reader in (rgb_reader, instance_reader):
                reader.attach(product)
                resources.callback(reader.detach, product)
            # Warm the newly attached graph, then fence completed deliveries.
            # These are render-only updates; no orchestrator/physics step.
            previous = None
            for _ in range(3):
                self._env.render()
                stamp = dispatcher_time()
                if stamp is None or not np.isfinite(stamp):
                    raise RuntimeError("No renderer delivery clock")
                if previous is not None and stamp <= previous:
                    raise RuntimeError("Renderer did not deliver a new frame")
                previous = stamp
            rgb = np.asarray(rgb_reader.get_data())
            segmentation = instance_reader.get_data()
            instances = np.asarray(segmentation["data"])
            if rgb.shape not in ((height, width, 3), (height, width, 4)):
                raise ValueError(f"Unexpected RGB shape: {rgb.shape}")
            if rgb.dtype != np.uint8:
                raise ValueError(f"Unexpected RGB dtype: {rgb.dtype}")
            if instances.shape == (height, width, 1):
                instances = instances[:, :, 0]
            if instances.shape != (height, width):
                raise ValueError(
                    f"Unexpected instance shape: {instances.shape}"
                )
            if instances.dtype != np.uint32:
                raise ValueError(
                    f"Unexpected instance dtype: {instances.dtype}"
                )
            if dispatcher_time() != previous:
                raise RuntimeError("Concurrent render during visual read")
            return (
                rgb[:, :, :3].copy(),
                instances.copy(),
                dict(segmentation["info"]["idToLabels"]),
                {
                    "render_reference_time": float(previous),
                    "render_passes": 3,
                    "camera_prim_path": str(camera.prim_path),
                },
            )

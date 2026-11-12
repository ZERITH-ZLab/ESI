"""Query live scene objects through one task-independent, JSON-safe interface.

Call on the simulator thread between physics steps. This module never steps,
renders, resets, loads assets, or commands the robot. It reads privileged state,
not an image-grounded observation. Resolve object IDs again after reset.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    """Convert native NumPy or Torch data to a finite CPU array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Native scene data contains NaN or infinity")
    return result


def _vector(value: Any, size: int) -> np.ndarray:
    result = _array(value).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"Expected {size} elements, got {result.shape}")
    return result


def _points(value: Any) -> np.ndarray:
    result = _array(value)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 point array, got {result.shape}")
    return result


def _json_value(value: Any) -> Any:
    """Copy native containers without mutating or stringifying unknown data."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Native scene data contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported native value: {type(value).__name__}")


def _available(value: Any, source: str, frame: str | None = None) -> dict:
    return {"available": True, "value": value, "source": source, "frame": frame}


def _unavailable(reason: str) -> dict:
    return {"available": False, "value": None, "reason": reason}


class SceneQuery:
    """Read objects by label or runtime instance ID, independent of task/type.

    Args:
        env: Initialized native RoboDojo environment, exposing scene_manager.

    All public operations use query(); fields select information, not an object
    category. Missing capabilities are explicit unavailable results. Invalid
    selectors, malformed state and provider failures raise rather than falling
    back to initial layout values. This is not an RPC server or a grasp planner.
    """

    FIELDS = (
        "pose",
        "description",
        "metadata",
        "functional_points",
        "joints",
        "mesh",
        "particles",
    )

    def __init__(self, env: Any):
        self._scene = env.scene_manager
        self._layout = self._scene.layout_manager

    def query(
        self,
        identifiers: str | Sequence[str] | None = None,
        *,
        fields: Sequence[str] = ("pose", "description"),
        env_idx: int = 0,
    ) -> dict:
        """Read selected scene-object fields without advancing simulation.

        Args:
            identifiers: Exact labels or instance IDs; None lists all registered
                task objects. No fuzzy matching. An empty list selects none.
            fields: Unique field names in FIELDS. Mesh/particles are opt-in and
                return all points, without sampling or truncation.
            env_idx: Zero-based native sub-environment index.

        Returns:
            JSON-safe dictionary with schema_version, env_idx, coordinate_frame,
            position_unit, quaternion_order, privileged_state, and objects.
            Objects contain id, label, model_name, model_id, physics_type,
            and fields.
            Each field reports available/value/source/frame, or available=False
            with a reason. Spatial query results use the sub-environment frame
            in meters and wxyz quaternions. Metadata remains unmodified asset
            metadata; joint positions/limits retain explicitly native units.

        Raises:
            KeyError: An identifier has no match.
            ValueError: Ambiguous/duplicate selectors, bad field/index or state.
            TypeError: Invalid argument types or non-serializable native data.
            RuntimeError: A registered object has no live scene handle.

        Serialize queries with simulation updates. No atomic cross-thread
        snapshot, timestamp, visibility or grasp success is assumed.
        """
        if (
            isinstance(env_idx, bool)
            or not isinstance(env_idx, Integral)
            or not 0 <= env_idx < len(self._scene.env_origins)
        ):
            raise ValueError("env_idx must index an existing sub-environment")
        if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
            raise TypeError("fields must be a sequence of field names")
        if any(
            not isinstance(field, str) or field not in self.FIELDS
            for field in fields
        ):
            raise ValueError(f"fields must be selected from {self.FIELDS}")
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate fields")
        records = self._records(int(env_idx))
        selected = self._select(records, identifiers)
        origin = _vector(self._scene.env_origins[env_idx], 3)
        objects = []
        for physics_type, record in selected:
            instance_id = record["inst_name"]
            obj = self._layout.get_scene_object(env_idx, instance_id)
            if obj is None:
                raise RuntimeError(f"No live scene handle for {instance_id}")
            metadata = self._layout.get_instance_metadata(
                env_idx, inst_name=instance_id
            )
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError(f"Invalid metadata for {instance_id}")
            cache = {"env_idx": int(env_idx), "instance_id": instance_id}
            result = {}
            for field in fields:
                reader = getattr(self, f"_read_{field}")
                result[field] = reader(obj, metadata, origin, cache)
            objects.append(
                {
                    "id": instance_id,
                    "label": record.get("label"),
                    "model_name": None
                    if metadata is None
                    else metadata.get("model_name"),
                    "model_id": None
                    if metadata is None
                    else metadata.get("model_id"),
                    "physics_type": physics_type,
                    "fields": result,
                }
            )
        return _json_value(
            {
                "schema_version": 1,
                "env_idx": int(env_idx),
                "coordinate_frame": "sub_environment",
                "position_unit": "m",
                "quaternion_order": "wxyz",
                "privileged_state": True,
                "objects": objects,
            }
        )

    def _records(self, env_idx):
        records = []
        seen = set()
        for (
            physics_type,
            collection,
        ) in self._layout.object_records_by_type.items():
            for record in collection.layout_records_by_env[env_idx]:
                instance_id = record["inst_name"]
                if (
                    not isinstance(instance_id, str)
                    or not instance_id
                    or instance_id in seen
                ):
                    raise ValueError(
                        f"Invalid or duplicate instance ID: {instance_id}"
                    )
                seen.add(instance_id)
                records.append((physics_type, record))
        return sorted(records, key=lambda item: item[1]["inst_name"])

    @staticmethod
    def _select(records, identifiers):
        if identifiers is None:
            return records
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        if not isinstance(identifiers, Sequence) or isinstance(
            identifiers, bytes
        ):
            raise TypeError("identifiers must be a label/ID string or sequence")
        selected = []
        seen = set()
        for identifier in identifiers:
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Each identifier must be a non-empty string")
            matches = [
                item
                for item in records
                if identifier in (item[1]["inst_name"], item[1].get("label"))
            ]
            if not matches:
                raise KeyError(f"Unknown object: {identifier}")
            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous object: {identifier}; use an instance ID"
                )
            instance_id = matches[0][1]["inst_name"]
            if instance_id in seen:
                raise ValueError(
                    f"Object selected more than once: {instance_id}"
                )
            seen.add(instance_id)
            selected.append(matches[0])
        return selected

    @staticmethod
    def _read_pose(obj, metadata, origin, cache):
        getter = getattr(obj, "get_world_pose", None)
        if not callable(getter):
            return _unavailable(
                "No native root-pose capability; not a particle centroid"
            )
        position, quaternion = getter()
        position = _vector(position, 3) - origin
        quaternion = _vector(quaternion, 4)
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3):
            raise ValueError("Native pose must contain a unit wxyz quaternion")
        return _available(
            {"position": position, "quaternion_wxyz": quaternion},
            "object.get_world_pose",
            "sub_environment",
        )

    @staticmethod
    def _read_description(obj, metadata, origin, cache):
        if metadata is None or not metadata.get("description"):
            return _unavailable(
                "No asset description; label is not a semantic description"
            )
        return _available(metadata["description"], "asset_metadata.description")

    @staticmethod
    def _read_metadata(obj, metadata, origin, cache):
        if metadata is None:
            return _unavailable("No asset metadata")
        return _available(
            metadata, "layout_manager.get_instance_metadata", "asset_native"
        )

    @staticmethod
    def _read_joints(obj, metadata, origin, cache):
        getter = getattr(obj, "get_all_joints_info", None)
        if not callable(getter):
            return _unavailable("No native joint-state capability")
        return _available(
            {
                "joints": getter(),
                "units": "provider_native",
                "note": "Unlabeled native joint types/units; not converted",
            },
            "object.get_all_joints_info",
        )

    @staticmethod
    def _read_mesh(obj, metadata, origin, cache):
        if "mesh" not in cache:
            getter = getattr(obj, "sample_mesh_vertices", None)
            if not callable(getter):
                cache["mesh"] = _unavailable("No native mesh-vertex capability")
            else:
                points = _array(getter()[0])
                if points.size == 0:
                    cache["mesh"] = _unavailable(
                        "Native mesh provider returned no vertices"
                    )
                else:
                    points = _points(points) - origin
                    cache["mesh"] = _available(
                        {
                            "positions": points,
                            "count": len(points),
                            "coverage": "May omit other submeshes",
                        },
                        "object.sample_mesh_vertices",
                        "sub_environment",
                    )
        return cache["mesh"]

    @staticmethod
    def _read_particles(obj, metadata, origin, cache):
        getter = getattr(obj, "get_particle_positions", None)
        if not callable(getter):
            return _unavailable("No native particle-position capability")
        points = _array(getter()[0])
        if points.size == 0:
            return _unavailable(
                "Native particle provider returned no particles"
            )
        points = _points(points)
        # RoboDojo returns PointInstancer-local positions, not world positions.
        # Apply the actual USD transform (including translation/rotation/scale).
        from pxr import Usd, UsdGeom

        prim = obj.point_instancer.GetPrim()
        if not prim.IsValid():
            raise ValueError("Invalid particle PointInstancer prim")
        transform = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ),
            dtype=float,
        )
        world = np.column_stack((points, np.ones(len(points)))) @ transform
        positions = _points(world[:, :3]) - origin
        return _available(
            {"positions": positions, "count": len(positions)},
            "object.get_particle_positions + USD local-to-world",
            "sub_environment",
        )

    def _read_functional_points(self, obj, metadata, origin, cache):
        if metadata is None:
            return _unavailable("No asset metadata")
        annotations = []
        for role in ("active", "passive"):
            for tag, annotation in (
                metadata.get(role, {}).get("functional", {}).items()
            ):
                item = {"role": role, "tag": tag, "annotation": annotation}
                if annotation.get("frame") is not None:
                    # Use the official transform, not inferred grasp poses.
                    frames = self._layout.get_functional_points(
                        tag=tag,
                        type=role,
                        config=metadata,
                        ret="list",
                        obj_name=cache["instance_id"],
                        env_idx=cache["env_idx"],
                    )
                    frames = [_vector(frame, 7) for frame in frames]
                    if any(
                        not np.isclose(
                            np.linalg.norm(frame[3:]), 1.0, atol=1e-3
                        )
                        for frame in frames
                    ):
                        raise ValueError(
                            f"Invalid functional-point quaternion: {tag}"
                        )
                    item["coordinates"] = _available(
                        frames,
                        "layout_manager.get_functional_points",
                        "sub_environment",
                    )
                elif annotation.get("id") is not None:
                    mesh = self._read_mesh(obj, metadata, origin, cache)
                    if not mesh["available"]:
                        item["coordinates"] = _unavailable(
                            "Mesh required for vertex-index annotation"
                        )
                    else:
                        indices = annotation["id"]
                        if (
                            not isinstance(indices, Sequence)
                            or not indices
                            or any(
                                isinstance(i, bool)
                                or not isinstance(i, Integral)
                                for i in indices
                            )
                        ):
                            raise ValueError(
                                f"Invalid functional-point indices: {tag}"
                            )
                        points = mesh["value"]["positions"]
                        if min(indices) < 0 or max(indices) >= len(points):
                            raise ValueError(
                                f"Functional-point index out of range: {tag}"
                            )
                        item["coordinates"] = _available(
                            points[list(indices)],
                            "mesh vertices selected by asset annotation",
                            "sub_environment",
                        )
                else:
                    item["coordinates"] = _unavailable(
                        "Annotation does not contain frame or vertex IDs"
                    )
                annotations.append(item)
        if not annotations:
            return _unavailable("No functional-point annotations")
        return _available(
            annotations, "asset functional annotations; NOT grasp candidates"
        )

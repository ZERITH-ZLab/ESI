"""Fixed task examples for GPT only; never simulator truth or pi05 observations."""
from pathlib import Path
import hashlib
import os
import shutil

from PIL import Image

from .io import write_json

REFERENCE_ROOT = Path(__file__).with_name("task_references")


def prepare_task_references(task, output):
    """Snapshot a supplied reference for collection and eval using the same path.

    No placeholder or synthetic success image is used when the asset is pending.
    Reattaching the fixed image with each RGB observation keeps it available after
    context compaction. The per-run copy also owns any resized GPT preview.
    """
    from .prompt_profile import guidance_disabled
    if guidance_disabled() or task != "store_tools_in_toolbox":
        return []
    output = Path(output)
    directory = output / "task_references"
    directory.mkdir(parents=True, exist_ok=True)
    enabled = os.environ.get("ROLLOUT_TASK_REFERENCE_IMAGES", "1")
    if enabled not in ("0", "1"):
        raise ValueError("ROLLOUT_TASK_REFERENCE_IMAGES must be 0 or 1")
    source = REFERENCE_ROOT / "store_tools_in_toolbox" / "completed.png"
    manifest = dict(task=task, version="fixed_task_reference.v1",
                    current_scene=False, consumer="gpt_only",
                    status="disabled" if enabled == "0" else "pending_image", images=[])
    if enabled == "1" and source.is_file():
        payload = source.read_bytes()
        with Image.open(source) as image:
            if image.format != "PNG":
                raise ValueError("Toolbox completed.png must be a real PNG")
            image.verify()
        path = directory / "toolbox_completed.png"
        shutil.copyfile(source, path)
        descriptor = dict(
            camera="task_reference_toolbox_completed", path=str(path),
            role="fixed_completed_state_example", current_scene=False,
            sha256=hashlib.sha256(payload).hexdigest(),
            caption="Fixed completed toolbox example, NOT the current observation. "
                    "Match slot shapes and tool orientation to the current toolbox; "
                    "do not copy absolute image coordinates.")
        manifest.update(status="available", images=[descriptor])
    write_json(directory / "manifest.json", manifest)
    return manifest["images"]


def append_reference_images(visible_packet, attachments, references, max_image_edge):
    """Append separately labelled references after the current-camera images."""
    import base64
    from .skill.image_preview import prepare_image
    if not references:
        return
    visible_packet["task_reference_images"] = []
    for item in references:
        descriptor, payload = prepare_image(item, max_image_edge)
        visible_packet["task_reference_images"].append(descriptor)
        attachments.append(dict(type="inputImage",
                                imageUrl="data:image/png;base64," +
                                base64.b64encode(payload).decode("ascii")))

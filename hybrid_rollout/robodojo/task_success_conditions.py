"""Append a concise task-completion brief to a GPT prompt.

Native checks are defined by the benchmark; only the selected
task_completion_briefs.md section is sent. This module never reads episode
layouts or simulator state, and never changes native success checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


CATALOG = Path(__file__).with_name("task_completion_briefs.md")
GUIDANCE = Path(__file__).with_name("manipulation_guidance.md")
_TASK_HEADING = re.compile(r"^## ([A-Za-z0-9_]+)$", re.MULTILINE)
_PROMPT_HEADING = "# Native task success conditions:"


def render_task_success_conditions(task: str, *, catalog: Path = CATALOG, catalog_text: str | None = None) -> str:
    """Return the exact task's brief, without evaluator implementation details.

    Args:
        task: Native runtime task name, including ``_random`` when applicable.
        catalog: Reviewed Markdown catalog to read.

    Returns:
        A prompt section containing only this task and relevant return semantics.

    Raises:
        ValueError: The task is unknown or the catalog has duplicate headings.
    """
    text = catalog.read_text(encoding="utf-8") if catalog_text is None else catalog_text
    headings = list(_TASK_HEADING.finditer(text))
    names = [heading.group(1) for heading in headings]
    if not headings or len(set(names)) != len(names):
        raise ValueError("Success-condition catalog has missing or duplicate task headings")
    if task not in names:
        raise ValueError(f"No reviewed native success conditions for task {task!r}")
    index = names.index(task)
    start = headings[index].start()
    end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
    brief = text[start:end].strip()
    # Expand only for tasks requiring a final return; pickup/insertion tasks
    # must not acquire an unrelated reset requirement from a common preamble.
    return_note = ""
    if "return the arms" in brief.lower():
        return_note = (
            "\n\nReturn your controlled arms to their recorded starting poses, "
            "not to an all-zero joint configuration. Do not command demonstration "
            "or other support arms."
        )
    return (
        f"{_PROMPT_HEADING} {task}\n\n"
        "These are task completion criteria, not live scene information. Use the "
        "current instruction and permitted observations to identify objects and "
        "assess progress; the simulator reports final success. Do not invent "
        "hidden object measurements. No extra per-step checklist or tool calls "
        "are required solely to restate these criteria.\n\n"
        + brief
        + return_note
        + "\n"
    )


def append_task_success_conditions(
    prompt: str, task: str, *, output_dir: Path | None = None
) -> str:
    """Append one task section, optionally saving the exact injected text.

    Args:
        prompt: Existing GPT developer prompt.
        task: Exact native runtime task name.
        output_dir: Existing audit directory in which to save the injected text.

    Returns:
        The existing prompt followed by the reviewed task section.

    Raises:
        ValueError: The prompt already contains a task-success section, or the
            requested task is absent from the reviewed catalog.
    """
    if _PROMPT_HEADING in prompt:
        raise ValueError("Task success conditions have already been appended")
    from .prompt_evolution.bundle import from_env, render
    bundle = from_env()
    if bundle is None:
        section = render_task_success_conditions(task)
        guidance = GUIDANCE.read_text(encoding="utf-8")
    else:
        guidance, brief = render(bundle, task)
        section = render_task_success_conditions(task, catalog_text=brief)
        if output_dir is not None:
            from .io import write_json
            write_json(output_dir / 'guidance_bundle.json', bundle)
    if output_dir is not None:
        (output_dir / "task_success_conditions.md").write_text(section, encoding="utf-8")
        (output_dir / "manipulation_guidance.md").write_text(guidance, encoding="utf-8")
    return prompt + "\n\n" + guidance + "\n\n" + section


def main() -> None:
    """Render one task's rules, or append them to an existing prompt file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Exact native runtime task name")
    parser.add_argument("--prompt", type=Path, help="Existing UTF-8 prompt file")
    parser.add_argument("--output", type=Path, help="Write the result instead of stdout")
    args = parser.parse_args()
    if args.prompt is None:
        result = render_task_success_conditions(args.task)
    else:
        result = append_task_success_conditions(
            args.prompt.read_text(encoding="utf-8"), args.task
        )
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()

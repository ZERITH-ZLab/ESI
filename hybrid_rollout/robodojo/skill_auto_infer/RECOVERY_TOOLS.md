# Motion tools

GPT corrections use the motion interfaces defined in `SKILL.md` and `context/eef_control.md`.
Requests are bound to the current decision and validated before execution. An execution failure can still have completed some actions; use the returned receipts to distinguish planning failure from partial execution.

RoboDojo-specific closing waits and grasp assistance are configured in `configs/environments/robodojo.json`. Keep these settings separate from task guidance. Motion and collision validation do not establish task success.

Execution receipts retain requested targets, executed actions and errors. Do not count requested waypoints as environment steps.

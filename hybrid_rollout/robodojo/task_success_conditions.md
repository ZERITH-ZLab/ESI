# Native task criteria

RoboDojo defines task success and scoring in its task and reward-manager source.
Use the versions documented in `docs/INSTALLATION.md` when checking those definitions.

`task_completion_briefs.md` supplies the controller-visible task descriptions.
The controller saves the selected description as `task_success_conditions.md` in each episode workspace.
Native evaluator results, not generated reasoning or tool completion, determine benchmark success.

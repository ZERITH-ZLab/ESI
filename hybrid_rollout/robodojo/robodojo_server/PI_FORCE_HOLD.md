# RoboDojo grasp assistance

Automatic hold and gripper adjustment are configured in `configs/environments/robodojo.json`.
These options depend on the RoboDojo X5 gripper mapping, contact feedback and control timing.
Do not reuse them in a different environment without adapting and validating those interfaces.

- `pi_force_hold.enabled=false` disables automatic hold and adjustment.
- `pi_force_hold.config.tighten_enabled=false` retains automatic hold without changing the predicted gripper target.
- GPT explicit closing waits are configured independently under `gpt_gripper_hold`.

Contact and position stability are execution feedback, not proof of task success.
Receipts preserve the original command, overrides, inserted holds and actual executed action.
Training exports use executed actions; unexecuted proposal tails are not supervision.

See [configuration](../../../docs/CONFIGURATION.md) for usage.

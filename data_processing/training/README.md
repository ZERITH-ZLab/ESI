# Training

Use `scripts/train.sh` from the repository root. The [workflow guide](../../docs/CONFIGURATION.md) covers data preparation, launch and resume commands.

- `configs/training/round*.json`: training mode, optimizer, update count and checkpoint settings.
- `configs/sampling.json`: task, controller, outcome and round sampling weights.
- `configs/deployment.local.json`: Python environment, paths, device count and parallelism.
- `configs/deployments/`: example deployment sizes.

The default mode is `subtask`; use `--set run.mode=task` for episode instructions only.
Prepare the training index and normalization statistics before launching. Preview the resolved configuration with `--show-config`.

Global batch size is `num_devices × per_device_batch_size × gradient_accumulation_steps`.
Launch one JAX process per node, not one per GPU. Nodes must use the same code, configuration and shared data paths.

For a short check, append `-- --stop-after 2`; resume with the same configuration and `-- --resume`.
Changing data or training configuration requires a new run name.

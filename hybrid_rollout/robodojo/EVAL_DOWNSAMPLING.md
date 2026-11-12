# Evaluation trajectory downsampling

The RoboDojo adapter settings are in `configs/environments/robodojo.json`. To disable downsampling for a new final evaluation, pass `--set robodojo.eval_downsampling.mode=off`. Execution receipts retain source indices and actual executed steps; use those receipts, not proposal length, when counting actions.

See [configuration](../../docs/CONFIGURATION.md) and [evaluation setup](../../docs/EVALUATION.md) for supported launch commands.

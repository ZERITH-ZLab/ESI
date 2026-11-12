# Dataset export

For the training workflow, use `scripts/data.sh` as described in [configuration](../../../docs/CONFIGURATION.md).
The lower-level exporter accepts an extraction index:

```bash
bash hybrid_rollout/robodojo/tools/export_datasets.sh \
  --index /mnt/esi/extractions/run1/index.json \
  --output /mnt/esi/exports/run1 \
  --action-label command --formats both
```

Use a new output directory. Export reads source trajectories without modifying them; check `export_manifest.json` before using the output. Set `EXPORT_PYTHON` and `EXPORT_DEPS` for a custom installation.

`state[t]` is the measured state before execution; `command[t]` is the executed command; `next_state[t]` is the measured state afterward. Training uses command actions. Control frequency comes from `control_dt`; requested actions and measured states are not interchangeable.

Outputs include LeRobot datasets, MCAP files and per-episode sidecars. MCAP retains T+1 observations for T actions and uses episode-relative simulator timestamps. Gripper values remain normalized quantities, not calibrated physical widths. Coordinate-frame and unit metadata must be checked before importing into another data system.

Exported data still requires annotation-based filtering, indexing and normalization. Keep source observations accessible when training, because indexes may reference their original locations.

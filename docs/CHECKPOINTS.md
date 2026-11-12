# Checkpoints

| Model | Download | Local reference path | Use |
| --- | --- | --- | --- |
| ESI final π05 | **Pending publication — URL and checksum to be added** | `/mnt/models/esi-final` | Default final ESI evaluation |
| RoboDojo pretrained π05 | [RoboDojo dataset repository](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) | `/mnt/models/robodojo_pi05/59999` | Bootstrap collection and baseline evaluation |

The final checkpoint download is not yet available. An authorized local copy is required until publication. The launcher does not substitute the pretrained model when the final checkpoint is missing.

The configured path must point to the checkpoint root, not its parent, a training experiment directory, or `params/` alone:

```text
/mnt/models/esi-final/
├── params/                          # complete Orbax parameter directory
└── assets/
    └── arx_x5_sim/
        └── norm_stats.json
```

Keep all files inside `params/`; its metadata and array shards are required together. Optimizer `train_state/` and training-progress files are not needed for inference. The normalizer shipped with the final checkpoint is required; do not replace it with the pretrained normalizer.

The final path is `paths.final_checkpoint` in `configs/deployment.local.json`; the official pretrained path is `paths.official_checkpoint`. The latter is not needed when running only final evaluation.

For the pretrained model, use RoboDojo's checkpoint download instructions and select `Pi_05`, `RoboDojo-sim-arx_x5-joint-0/59999`. This is a separate artifact from ESI final.

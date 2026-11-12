# Checkpoints

| Model | Download | Local reference path | Use |
| --- | --- | --- | --- |
| ESI final π05 | [ESI-pi05](https://huggingface.co/Embodied-Self-Improvement/ESI-pi05) | `/mnt/models/esi-final` | Default final ESI evaluation |
| RoboDojo pretrained π05 | [RoboDojo dataset repository](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) | `/mnt/models/robodojo_pi05/59999` | Bootstrap collection and baseline evaluation |

## Download ESI final

The public model repository can be downloaded without authentication. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) in a separate tools environment if it is not already available. The following command downloads the release revision (21 parameter and normalization files, approximately 11.6 GiB):

```bash
hf download Embodied-Self-Improvement/ESI-pi05 \
  --revision 09395c2dd983b3f4d9dbd877297a2eecfd31ef6e \
  --include "params/**" "assets/**" \
  --local-dir /mnt/models/esi-final
```

Use a writable directory with sufficient free space. If a download is interrupted, rerun the same command; retain the local download metadata so that up-to-date files can be reused. The pinned revision prevents later repository updates from changing the selected weights.

From the ESI source repository root, verify the downloaded files:

```bash
ESI_SOURCE_ROOT="$PWD"
(
  cd /mnt/models/esi-final
  sha256sum --check "$ESI_SOURCE_ROOT/configs/checkpoints/esi-final.sha256"
)
```

All 21 entries should report `OK`. The checksum list applies to the pinned revision above.

## Configure the checkpoint path

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

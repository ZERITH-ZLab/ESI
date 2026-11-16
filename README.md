<p align="right">
  <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
</p>

<h1 align="center">ESI: An Embodied Self-Improvement Flywheel for LLM-VLA Agents</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2611.18176"><img src="https://img.shields.io/badge/arXiv-2611.18176-b31b1b?style=flat&amp;labelColor=555" alt="arXiv 2611.18176"></a>
  &nbsp;
  <a href="https://www.zerith.com/blog/esi/"><img src="https://img.shields.io/badge/Project_Page-green?style=flat" alt="Project Page"></a>
</p>

ESI combines LLM planning and correction with VLA action execution, improving their collaboration through interaction with the environment. The flywheel follows **collection → subtask annotation → data preparation → training → next-round collection**, with optional prompt updates based on collected experience.

<p align="center">
  <a href="https://www.zerith.com/blog/esi/">
    <img src="assets/esi_overview.png" width="780" alt="The ESI flywheel: LLM–VLA agents collect experience, update guidance and policies, and begin the next round of interaction.">
  </a>
</p>

<p align="center"><em>Figure 1 from the paper: from collaborative execution to learning from experience. Click the image for interactive demonstrations and rollout videos.</em></p>

## Deployment

This repository provides deployment and evaluation support for the RoboDojo simulator. External projects, assets, and models must be installed or downloaded separately; see the [installation guide](docs/INSTALLATION.md) (Chinese) for dependency versions and required patches.

| Component | Recommended setup |
| --- | --- |
| OS and CUDA | Linux x86_64, Ubuntu 22.04, CUDA Toolkit 12.8; use a `devel` image for containers |
| Python | Python 3.11, with separate environments for RoboDojo and OpenPI |
| Simulator | Isaac Sim 5.1.0 with the specified versions of RoboDojo, XPolicyLab, IsaacLab, and cuRobo |
| Hardware | An RTX rendering GPU (e.g., RTX 4090) for simulation; VLA inference/training may use a separate compute GPU |

Recommended directory layout is shown below. Paths are configurable; preserve the internal directory structure of external projects:

```text
/opt/
├── ESI/                                      # This repository
├── RoboDojo/
│   ├── Assets -> /mnt/robodojo/Assets
│   ├── XPolicyLab/policy/Pi_05/openpi/        # OpenPI source and .venv
│   └── third_party/                          # IsaacLab and cuRobo
└── envs/RoboDojo/                            # Simulator and control-tools environment
/mnt/
├── robodojo/Assets/                          # Simulation assets
├── models/esi-final/                         # ESI model
├── esi/                                     # Trajectories, annotations, training and evaluation outputs
└── esi-private/                             # Account credentials
```

Deploy on a single machine or run the simulator and policy service on separate machines. The launcher starts the simulator over SSH, including a connection to the local machine for single-machine deployments. Multi-machine deployments must access the same shared output storage at the same absolute path on both sides. Compute GPUs such as A100/H100 cannot replace the RTX rendering GPU required by the simulator.

## Project structure and configuration

| Directory | Purpose |
| --- | --- |
| `scripts/` | Entry points for collection, annotation, data preparation, training, and evaluation |
| `configs/` | Deployment parameters, model settings, and stage recipes |
| `hybrid_rollout/robodojo/` | LLM–VLA control flow, simulator and policy services, prompts, and skills |
| `data_processing/` | Subtask annotation, training indexes, sampling, and training |
| `patches/` | Integration patches for external dependencies |

Start with the single-worker template. Run from the repository root, preserving any existing local configuration:

```bash
cp -n configs/deployment.eval.example.json configs/deployment.local.json
export ESI_PYTHON=/opt/envs/RoboDojo/bin/python
```

Set paths, SSH users/hosts, GPUs, concurrency, and the account-pool location in `configs/deployment.local.json`, then complete account login. LLM settings are in `configs/llm.json`, tasks in `configs/tasks.json`, and training parameters in `configs/training/`; other parameters can use the recipe defaults. See the [configuration reference](docs/CONFIGURATION.md) (Chinese) for account setup and commands for each stage.

## ESI evaluation

Download the [ESI-pi05 model](https://huggingface.co/Embodied-Self-Improvement/ESI-pi05) (approximately 11.6 GiB):

```bash
hf download Embodied-Self-Improvement/ESI-pi05 \
  --revision 09395c2dd983b3f4d9dbd877297a2eecfd31ef6e \
  --include "params/**" "assets/**" \
  --local-dir /mnt/models/esi-final
```

Set `paths.final_checkpoint` to the downloaded directory containing `params/` and `assets/`. Once the environment and accounts are configured, launch evaluation:

```bash
bash scripts/eval.sh --set run.name=esi_eval --show-config
bash scripts/eval.sh --set run.name=esi_eval
```

By default, evaluation uses the released ESI model with GPT subtask guidance on 10 tasks with 20 cases each, using the benchmark’s native budgets and scoring. The launcher starts the required services automatically; `--show-config` only previews the configuration.

For baseline evaluation, use `--config configs/eval/baseline.json`, prepare the RoboDojo pretrained weights separately, and set `paths.official_checkpoint`. Run long jobs in tmux. To resume, reuse the original command and configuration; choose a new `run.name` when changing experiment settings.

## Flywheel training

Each stage can be launched independently, using outputs from the preceding stage:

| Stage | Entry point | Input → Output |
| --- | --- | --- |
| Collection | `scripts/collect.sh` | Model and tasks → interaction trajectories |
| Annotation | `scripts/annotate.sh` | Trajectories → subtask and outcome annotations |
| Data preparation | `scripts/data.sh` | Trajectories and annotations → exported data, training indexes, and normalization statistics |
| Training | `scripts/train.sh` | Data index and model → updated checkpoint |
| Optional prompt update | `scripts/evolve.sh` | Collected experience → prompt bundle for subsequent runs |

Use `configs/collection/bootstrap.json` and `configs/annotation/offline.json` for the first round. For subsequent rounds, use `feedforward.json` and `online.json`, setting the new checkpoint as the model for the next collection round. Training defaults to the `subtask` recipe, with an explicit `task` option available; automatic prompt updates are disabled by default. See the configuration reference above for stage parameters, data paths, and multi-GPU deployment.

## License

Original ESI contributions are licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

Inherited code and third-party components retain their original licenses. See [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

If you use ESI in your research, please cite our [arXiv technical report](https://arxiv.org/abs/2611.18176):

```bibtex
@misc{cai2026esi,
  title = {ESI: An Embodied Self-Improvement Flywheel for LLM-VLA Agents},
  author = {Cai, Yundi and Wang, Zihan and Han, Chenxi and Cheng, Yi and Liu, Houde and Wang, Zilei},
  year = {2026},
  eprint = {2611.18176},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2611.18176}
}
```

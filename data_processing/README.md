# Data preparation

Run the stages from the repository root using the [workflow guide](../docs/CONFIGURATION.md).

| Stage | Entrypoint | Configuration |
| --- | --- | --- |
| Annotate trajectories | `scripts/annotate.sh` | `configs/annotation/offline.json` or `online.json` |
| Export, index, merge and normalize | `scripts/data.sh` | `configs/data.json` |
| Sample and train | `scripts/train.sh` | `configs/sampling.json`, `configs/training/round*.json` |

Offline annotation segments trajectories without online subtask inputs. Online annotation preserves the instructions actually supplied to the policy and assesses outcomes. Inspect validated annotations before preparing training data.

Exports retain executed actions and observation references. Keep the source observations, exports, indexes and normalization files accessible on training nodes; an index alone is not a self-contained dataset.

See [online annotation](online_subtask_outcomes/README.md), [training](training/README.md) and [visualization](policy_review/README.md) for component usage.

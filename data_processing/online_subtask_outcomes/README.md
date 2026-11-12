# Online subtask outcome annotation

Use this route for collection with GPT feedforward instructions. It preserves the actual online instruction, including null, and labels outcomes and instruction–action consistency. It does not resegment the trajectory or rewrite the instruction.

Start with `scripts/annotate.sh --config configs/annotation/online.json`; the [workflow guide](../../docs/CONFIGURATION.md) documents prepare, run and validate commands. Use the offline recipe for trajectories without online subtask inputs.

Labels cover executed decision intervals. Automatic grasp holds belong to the existing operation; they do not establish grasp success. Zero-step failures remain context without supervised actions. Outcomes distinguish success, failure, unknown and interrupted; evidence distinguishes direct confirmation from inference.

Inspect `latest_run.json` to locate the run. Formal outputs are `runs/<id>/episodes/<key>/annotation.json`; drafts and validation errors are retained separately. `episodes.index.csv` maps annotations to source trajectories. A failed annotation is not a published result.

Rerun the same command to resume; validated episodes are skipped. Inspect review flags before training. Source records remain unchanged. [Policy visualization](../policy_review/README.md) can display the online instructions and validated outcome labels together.

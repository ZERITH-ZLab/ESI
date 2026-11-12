# Optional prompt evolution

Automatic prompt evolution is disabled by default. Configure it in `configs/prompt_evolution.json`; account settings come from `deployment.annotation`.

Run it after collection using the [workflow guide](../../../docs/CONFIGURATION.md), or analyze a completed suite explicitly:

```bash
bash scripts/evolve.sh --set run.name=prompt_review \
  --set run.suite=/mnt/esi/collection_suites/round1_collection
```

Append `-- --prepare-only` to inspect inputs without calling the model. Evaluation data is not accepted as optimization input.

One persistent analysis session investigates evidence and proposes guidance changes. A separate persistent review session checks the proposals against source records. Evidence is read on demand; image and context budgets are configurable.

The process targets difficult-to-recover planning failures. It does not modify task definitions, control interfaces, model weights or simulator parameters. Proposed changes require evidence and review; unsupported changes remain drafts.

Source prompts and trajectories are read-only. Approved changes produce a separate bundle for a subsequent collection run. Select that bundle explicitly through `config.prompt_bundle` and `config.prompt_bundle_sha256`, using a new experiment name.

Progress and results are stored under the run's `analysis/` directory:

- `status.json`: current stage and progress.
- `lead/`, `reviewer/`: session records and model calls.
- `usage_summary.json`: reported token usage and missing-usage counts.
- `draft_bundle.json`, `patch_reviews.json`: proposals and review results.
- `candidate_bundle.json`: the approved bundle, when available.

Restart with the same configuration to resume. Keep the managed account profile and session records available; do not replace authentication directories during a run.

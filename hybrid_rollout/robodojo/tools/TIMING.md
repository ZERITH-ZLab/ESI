# Runtime timing

Each episode stores `timing/controller.jsonl` events and `timing/summary.json` aggregates.
Use them to distinguish controller, inference, execution and retry time.

- Teacher intervals include communication and tool waiting; they are not pure model compute time.
- RPC and native-tool intervals may be nested in controller intervals; do not add nested durations twice.
- `measurement_complete=false` indicates missing or truncated measurements.
- Open spans and missing values must not be treated as zero duration.
- Execution counts come from receipts; tool requests may fail or execute only partially.

Read timing files without modifying archived records. For offline analysis, write derived summaries to a separate output directory.

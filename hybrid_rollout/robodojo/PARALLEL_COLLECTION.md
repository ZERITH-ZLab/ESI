# Collection concurrency

Set `collection.workers` and `collection.slots` in the deployment configuration. Actual concurrency is bounded by available hardware and managed accounts. During a run, use the worker control file reported by the scheduler to adjust `desired_workers` within the configured slot count. Zero stops new dispatch while active episodes finish; it does not cancel their work.

See [configuration](../../docs/CONFIGURATION.md) and [evaluation setup](../../docs/EVALUATION.md) for supported launch commands.

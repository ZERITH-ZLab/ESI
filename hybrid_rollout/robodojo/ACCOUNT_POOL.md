# Managed account pool

For first deployment, follow [evaluation setup](../../docs/EVALUATION.md). Configure
`paths.auth_root`, `paths.codex_bin`, `paths.account_pool_file` and
`collection.auth_profile` in `configs/deployment.local.json`; do not edit launch
scripts or copy another user's credentials.

A minimal one-worker pool is:

```json
{
  "profiles": ["codex_a"],
  "check_interval_seconds": 60,
  "quota_recheck_seconds": 600
}
```

The profile must already have an independent managed login created by
`python -m hybrid_rollout.robodojo.codex_backend.profiles --shared-root <auth_root> login codex_a --codex <codex_bin>`.
`status` displays the credential locations. The default developer Codex login is
not implicitly used. Registered profile names are defined in `codex_backend/profiles.py`.

For explicit per-account limits use `accounts` instead of `profiles`:

```json
{
  "accounts": [
    {"name": "primary_account", "profiles": ["codex_a", "codex_a_2"], "max_concurrent": 2, "priority": "primary"},
    {"name": "backup_account", "profiles": ["codex_b"], "max_concurrent": 1, "priority": "backup"}
  ],
  "check_interval_seconds": 60,
  "quota_recheck_seconds": 600
}
```

Each group represents one real account. Each profile needs its own login; copying
refresh tokens is not a valid second slot. `max_concurrent` is the allowed number
of concurrent jobs, not a registration count. It must not exceed the number of
profiles in the group or eight. Hardware worker capacity is configured separately
under `collection`; account availability can reduce actual concurrency.

The pool uses OS locks and account identity checks. It never purchases quota or
logs in automatically. Unavailable accounts produce `waiting_for_accounts` with
per-profile reasons. A quota lookup failure does not prove an account is exhausted.
A normal native task failure is a completed evaluation, not grounds for retrying
until success. Recoverable infrastructure failures follow the queue's bounded
retry policy; incomplete episodes restart from the beginning, not from a saved
simulator state.

Existing registered slots can be added/removed through the pool JSON. Introducing
new names in the Python registry requires restarting the dispatcher after draining
its running episodes. Do not delete lock files to force-release accounts.

Pool progress is under the task result directory's `_account_pool/run_*/pool_run.json`.
Credentials and pool locks live under `auth_root`, outside the repository. Keep
permissions private. Annotation and optional prompt evolution select profiles via
`deployment.annotation`; they do not automatically use all collection slots.

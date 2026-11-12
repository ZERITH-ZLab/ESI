# Optional progress notifications

The watcher can send experiment progress to an ntfy topic. It is independent of rollout execution.
Run from the repository root:

```bash
python hybrid_rollout/robodojo/tools/watch_experiment.py --init
python hybrid_rollout/robodojo/tools/watch_experiment.py --help
```

The initialization command stores the topic and optional token in `~/.config/gpt-as-policy/ntfy.json` outside Git. Use a trusted server because notifications contain experiment progress.

Use `--once` to inspect a local summary without sending a notification. Stopping the watcher does not stop collection. A completed run is not necessarily a successful task.

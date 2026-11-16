# Run the final ESI evaluation

Complete [installation](INSTALLATION.md) and [checkpoint preparation](CHECKPOINTS.md) first. Run the commands below from the ESI repository root on the controller/policy node.

## Configure the deployment

```bash
cp configs/deployment.eval.example.json configs/deployment.local.json
```

The copied file inherits `deployment.example.json` and selects one worker for initial deployment. Edit `deployment.local.json`: override the `paths` entries that differ from the reference directory tree, especially `sim_ssh_user`, `sim_ssh_key`, `codex_bin` and `account_pool_file`; replace the simulator hostname and choose the GPU IDs/ports in `collection.slots`. To override a path, add a `paths` object containing that key. It is merged with the example, so unchanged defaults need not be copied.

`paths.sim_ssh_user` is the default remote username (root only as a compatibility default). An individual slot may override it with `ROLLOUT_SIM_SSH_USER`. This applies to remote checks, simulator lifecycle and SSH tunnels. The user must have access to the GPU and read/write permissions on the experiment directory; root privileges are not required by the launcher.

Set `paths.output_root` to a writable absolute directory outside the source repository, such as `/data/esi` or `/home/user/esi-output`; `/mnt` is not required. Use the same absolute path for shared experiment storage on all participating nodes. The controller writes a source snapshot there and the simulator runs that snapshot. A directory with the same name but different contents is not shared storage. Before loading services, the launcher checks that a temporary file can be read and updated remotely and that the update is visible locally. For a single-machine installation, configure SSH to that machine and budget GPU memory for both services. Check SSH key login before starting:

```bash
ssh -i /path/to/key -p 22 your_user@your_simulator_host 'hostname'
```

For more workers, add independent slots and increase `collection.workers`; adjust ports, GPU placement and account capacity together. The eight-slot example is a deployment preset, not a promise that eight instances fit on any GPU.

## Configure a managed account

Install a Codex CLI supporting the app-server interface and set `paths.codex_bin` to its executable. See [Codex CLI](https://github.com/openai/codex) for its installation instructions.

Choose one registered profile, for example `codex_a`, and log in using the project's managed home. `--shared-root` here must equal `paths.auth_root`, not the experiment output root:

```bash
export ESI_PYTHON=/opt/envs/RoboDojo/bin/python
"$ESI_PYTHON" -m hybrid_rollout.robodojo.codex_backend.profiles \
  --shared-root /mnt/esi-private login codex_a --codex /usr/local/bin/codex
"$ESI_PYTHON" -m hybrid_rollout.robodojo.codex_backend.profiles \
  --shared-root /mnt/esi-private status
```

Replace those paths if different. The managed login is separate from the developer's default Codex home. A working default CLI login is not automatically a managed profile login.

Create a private account-pool JSON and point `paths.account_pool_file` to it:

```json
{
  "profiles": ["codex_a"],
  "check_interval_seconds": 60,
  "quota_recheck_seconds": 600
}
```

Use `collection.auth_profile=codex_a`. One logged-in profile is enough for the one-worker preset. More workers need independently logged-in slots and the corresponding account concurrency settings. See [account pool](../hybrid_rollout/robodojo/ACCOUNT_POOL.md).

Configure any required proxy in the shell running the evaluation. The project does not install a proxy or manage subscriptions. Keep local simulator/policy RPC reachable; do not route localhost through the external proxy.

## Launch and resume

The default recipe is `configs/eval/final.json`. It already specifies the final policy route, prompts, skills and execution settings; no algorithm flags need to be assembled by hand. GPT defaults are in `configs/llm.json`; the account must support the configured model.

```bash
export ESI_PYTHON=/opt/envs/RoboDojo/bin/python
bash scripts/eval.sh --set run.name=esi_final_eval --show-config
# Start inside a tmux terminal, with any required proxy already configured.
bash scripts/eval.sh --set run.name=esi_final_eval
```

The model root defaults to `/mnt/models/esi-final` via `paths.final_checkpoint`. Missing weights cause an error, not a fallback to the official checkpoint. The default panel is 200 cases, with no inherited exclusions. Plan/schema/SSH tests are not a substitute for a completed evaluation.

Outputs are under `paths.output_root`, including `release_runs/esi_final_eval/eval/` for resolved configuration. The native queue prints its result and progress paths. Keep the source snapshot and raw results for resume and statistics. Repeat the same command to resume the same experiment; change `run.name` when changing model, prompt, paths or recipe. For a new experimental repeat, always use a new name.

## Baseline and training

The baseline is explicit and uses the separate official pretrained checkpoint:

```bash
bash scripts/eval.sh --config configs/eval/baseline.json --set run.name=baseline_eval
```

It does not inherit final ESI weights or GPT subtask guidance. The full flywheel and training configuration are described in [CONFIGURATION.md](CONFIGURATION.md); training is not required to evaluate an available final checkpoint.

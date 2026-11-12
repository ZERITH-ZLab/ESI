#!/usr/bin/env bash
# A100: bash hybrid_rollout/robodojo/launch_parallel_collection.sh [--dry-run|resume|resume-upgrade|--workers N]
set -euo pipefail
# New strategy/configuration => new experiment name. Same version restart => resume.
export ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:-arrange_largest_number_collect20x5_dynamic_v1}"
export ROLLOUT_PARALLEL_WORKERS=2
# 固定8槽位：0/1用两机GPU0，2/3用两机GPU1；4..7为对应GPU的第二个模拟器。
# 初次默认2路；运行时 --workers N 修改目标数量。A100端口从基数递增。
export ROLLOUT_SIM_SSH_PORT=4205 SIM_GPU=0 POLICY_GPU=0
export POLICY_PORT=18830 SIM_PORT=19113 ROLLOUT_SIM_REMOTE_PORT=19113
export ROLLOUT_SECOND_SIM_SSH_PORT=5727
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
export ROLLOUT_WORKER_CONTROL_FILE="${ROLLOUT_WORKER_CONTROL_FILE:-$ROLLOUT_SHARED_ROOT/results/$ROLLOUT_EXPERIMENT_ID/worker_control.json}"
export PYTHONPATH="$(cd "$ENTRY_ROOT/../.." && pwd)"
CONTROL_PYTHON="${ROBODOJO_PYTHON:-/opt/envs/RoboDojo/bin/python}"
if [[ "${1:-}" == --workers ]]; then
    [[ $# == 2 ]] || { echo 'Usage: launch_parallel_collection.sh --workers 0..8' >&2; exit 2; }
    exec "$CONTROL_PYTHON" -m hybrid_rollout.robodojo.parallel_collection \
        --config "$ROLLOUT_WORKER_CONTROL_FILE" --set-workers "$2"
fi
if [[ "${1:-}" != --dry-run ]]; then
    "$CONTROL_PYTHON" -m hybrid_rollout.robodojo.parallel_collection --config "$ROLLOUT_WORKER_CONTROL_FILE" --init
fi
exec bash "$ENTRY_ROOT/launch_collection.sh" "$@"

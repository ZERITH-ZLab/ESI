#!/usr/bin/env bash
# One isolated A100-controller + 4090-simulator episode.
set -euo pipefail
case "${ROLLOUT_CONTROLLER_ROUTE:-baseline}" in
    baseline) CONTROLLER_MODULE=hybrid_rollout.robodojo.skill.run ;;
    auto_infer_v1|subtask_feedforward_v1) CONTROLLER_MODULE=hybrid_rollout.robodojo.skill_auto_infer.run ;;
    *) echo 'Unknown ROLLOUT_CONTROLLER_ROUTE' >&2; exit 2 ;;
esac
if [[ "${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}" == pi05_only ]]; then
    CONTROLLER_MODULE=hybrid_rollout.robodojo.flywheel.student_eval
fi
case "${ROLLOUT_PI05_COLLISION_SHADOW:-0}" in
    0|1) ;;
    *) echo 'Invalid pi05 collision shadow flag' >&2; exit 2 ;;
esac
QUERY_ARGS=()
case "${ROLLOUT_ENABLE_MOTION_TOOLS:-0}" in
    0) ;;
    1) [[ "${ROLLOUT_CONTROLLER_ROUTE:-baseline}" == auto_infer_v1 || "${ROLLOUT_CONTROLLER_ROUTE:-baseline}" == subtask_feedforward_v1 ]] || { echo 'Motion tools require auto_infer_v1/subtask_feedforward_v1' >&2; exit 2; } ;;
    *) echo 'Invalid motion tools flag' >&2; exit 2 ;;
esac
case "${ROLLOUT_ENABLE_SCENE_QUERY:-0}" in
    0) ;;
    1) [[ "${ROLLOUT_CONTROLLER_ROUTE:-baseline}" == auto_infer_v1 || "${ROLLOUT_CONTROLLER_ROUTE:-baseline}" == subtask_feedforward_v1 ]] || { echo 'Scene query requires auto_infer_v1/subtask_feedforward_v1' >&2; exit 2; }
       QUERY_ARGS=(--enable-scene-query) ;;
    *) echo 'Invalid scene query flag' >&2; exit 2 ;;
esac
: "${CODE_ROOT:?}"; : "${ROBODOJO_PYTHON:?}"; : "${OPENPI_PYTHON:?}"
: "${OPENPI_SOURCE:?}"; : "${CHECKPOINT:?}"; : "${CODEX_BIN:?}"
: "${ROLLOUT_ARCHIVE:?}"; : "${ROLLOUT_CASE_FILE:?}"; : "${ROLLOUT_EVAL_MANIFEST:?}"
: "${ROBODOJO_TASK:?}"; : "${ROLLOUT_EVAL_SEED:?}"; : "${ROLLOUT_LAYOUT_ID:?}"
REMOTE_HOST="${ROLLOUT_SIM_HOST:?Set 4090 host}"
REMOTE_PORT="${ROLLOUT_SIM_REMOTE_PORT:-19113}"
LOCAL_PORT="${SIM_PORT:-19113}"
SSH_PORT="${ROLLOUT_SIM_SSH_PORT:-4205}"
SSH_KEY="${ROLLOUT_SIM_SSH_KEY:?Set simulator SSH key}"
SSH_USER="${ROLLOUT_SIM_SSH_USER:-root}"
if [[ ! "$SSH_USER" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*\$?$ ]]; then
    echo 'Invalid simulator SSH username' >&2; exit 2
fi
env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.deployment
CODE_ROOT="$ROLLOUT_ARCHIVE/source_snapshot"
REMOTE_ROOT="$CODE_ROOT"
cd "$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
REMOTE_ROBODOJO="${ROLLOUT_SIM_ROBODOJO_SOURCE:-/opt/RoboDojo}"
REMOTE_PYTHON="${ROLLOUT_SIM_PYTHON:-/opt/envs/RoboDojo/bin/python}"
SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
RUN_ROOT="$ROLLOUT_ARCHIVE/dual_machine"
STARTUP_TIMEOUT="${ROLLOUT_STARTUP_TIMEOUT:-900}"
# batch.py copies the frozen panel beside case_NNN.json.  Use that shared copy
# on 4090 rather than assuming the A100 repository path exists remotely.
REMOTE_MANIFEST="$ROLLOUT_ARCHIVE/evaluation_manifest.json"
REMOTE_PID="$RUN_ROOT/simulator.pid"; REMOTE_LOG="$RUN_ROOT/simulator.log"
TUNNEL_PID=''; POLICY_PID=''; CONTROLLER_PID=''; SIM_REQUESTED=0
source "$CODE_ROOT/hybrid_rollout/robodojo/pi05_load_lock.sh"
q() { printf '%q' "$1"; }
ssh_cmd() { ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
    -p "$SSH_PORT" "$SSH_USER@$REMOTE_HOST" "$@"; }
remote_alive() {
    ssh_cmd bash "$(q "$REMOTE_ROOT/hybrid_rollout/robodojo/remote_simulator.sh")" status "$(q "$REMOTE_PID")" \
        >/dev/null 2>&1
}
stop_local() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep .5; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    status=$?; trap - EXIT INT TERM; set +e
    stop_local "$CONTROLLER_PID"
    stop_local "$POLICY_PID"
    pi05_load_release
    stop_local "$TUNNEL_PID"
    # Stop the remote owner explicitly; closing SSH must not leave Isaac Sim alive.
    remote_status=0
    if (( SIM_REQUESTED )); then
        ssh_cmd bash "$(q "$REMOTE_ROOT/hybrid_rollout/robodojo/remote_simulator.sh")" stop "$(q "$REMOTE_PID")" "$REMOTE_PORT"
        remote_status=$?
    fi
    printf '%s\n' "$remote_status" > "$RUN_ROOT/remote_cleanup_status.txt"
    if (( remote_status != 0 )); then
        echo 'Remote cleanup NOT confirmed; inspect simulator.log before retrying' >&2
        (( status != 0 )) || status=5
    fi
    printf '%s\n' "$status" > "$RUN_ROOT/episode_exit_status.txt"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The case and manifest are on the shared CFS mount; the remote launcher reads
# the same immutable files and writes sim evidence into the same archive.
remote_start=(bash "$REMOTE_ROOT/hybrid_rollout/robodojo/remote_simulator.sh" start
  "$ROBODOJO_TASK" "$ROLLOUT_EVAL_SEED" "$REMOTE_PORT" "$ROLLOUT_ARCHIVE/sim"
  "$REMOTE_MANIFEST" "$ROLLOUT_ARCHIVE/evaluation_case.json" "$REMOTE_ROBODOJO" "$REMOTE_PYTHON"
  "$REMOTE_ROOT" "$REMOTE_PID" "$REMOTE_LOG" "${SIM_GPU:-0}" "${ROLLOUT_SIM_ENV_SCRIPT:-}" "${ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY:-0}" "${ROLLOUT_ENABLE_SCENE_QUERY:-0}" "${ROLLOUT_ENABLE_MOTION_TOOLS:-0}" "${ROLLOUT_COLLECTION_STEP_LIMIT:-0}" "${ROLLOUT_PI05_COLLISION_SHADOW:-0}"
  "${ROBODOJO_GRIPPER_CONTACT_ENABLED:-}" "${ROBODOJO_PI_FORCE_HOLD_ENABLED:-}" "${ROBODOJO_PI_FORCE_HOLD_CONFIG:-}" "${ROBODOJO_GRIPPER_HOLD_CONFIG:-}")
echo "Starting simulator: $REMOTE_LOG"
SIM_REQUESTED=1
ssh_cmd "$(printf '%q ' "${remote_start[@]}")"
deadline=$((SECONDS+STARTUP_TIMEOUT))
until grep -q '"event": "ready"' "$REMOTE_LOG" 2>/dev/null; do
    remote_alive || { tail -80 "$REMOTE_LOG" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo 'remote simulator startup timeout' >&2; exit 3; }
    sleep 2
done

pi05_load_acquire
echo "Starting pi05: $ROLLOUT_ARCHIVE/logs/policy.log"
env CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 PYTHONPATH="$CODE_ROOT:$OPENPI_SOURCE/src:$OPENPI_SOURCE/packages/openpi-client/src" \
    "$OPENPI_PYTHON" -m hybrid_rollout.robodojo.pi05_server.server \
    --checkpoint "$CHECKPOINT" --port "${POLICY_PORT:-18830}" \
    --policy-rng-seed "${ROLLOUT_POLICY_RNG_SEED:-0}" \
    --identity-output "$ROLLOUT_ARCHIVE/policy_identity.json" >"$ROLLOUT_ARCHIVE/logs/policy.log" 2>&1 & POLICY_PID=$!
deadline=$((SECONDS+STARTUP_TIMEOUT))
until [[ -f "$ROLLOUT_ARCHIVE/policy_identity.json" ]]; do
    kill -0 "$POLICY_PID" 2>/dev/null || { tail -80 "$ROLLOUT_ARCHIVE/logs/policy.log" >&2; exit 3; }
    (( SECONDS < deadline )) || { echo 'Policy startup timeout' >&2; exit 3; }
    sleep 2
done

env PYTHONPATH="$CODE_ROOT" "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.deployment verify-policy
pi05_load_release
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -p "$SSH_PORT" -N -T \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$SSH_USER@$REMOTE_HOST" & TUNNEL_PID=$!
sleep 2; kill -0 "$TUNNEL_PID" 2>/dev/null || exit 3

export CODEX_HOME="${ROLLOUT_CODEX_HOME_DIR:-${SHARED_ROOT}/private/auth_profiles/${ROLLOUT_AUTH_PROFILE:-codex_a}/codex_home}"
export ROLLOUT_CODEX_STATE_DIR="$SHARED_ROOT/private_runtime/$ROLLOUT_EXPERIMENT_ID/replica_$ROLLOUT_REPLICA_ID/attempt_$ROLLOUT_ATTEMPT/case_$ROLLOUT_INDEX/codex_state"
mkdir -p "$ROLLOUT_CODEX_STATE_DIR"
echo "Starting controller: $ROLLOUT_ARCHIVE/logs/controller.log"
env PYTHONPATH="$CODE_ROOT:$OPENPI_SOURCE/packages/openpi-client/src" \
    NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
    "$ROBODOJO_PYTHON" -u -m "$CONTROLLER_MODULE" \
    --output "$ROLLOUT_ARCHIVE/controller" --task "$ROBODOJO_TASK" \
    --evaluation-method "${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}" \
    --checkpoint "$CHECKPOINT" --student-port "${POLICY_PORT:-18830}" \
    --sim-port "$LOCAL_PORT" --codex "$CODEX_BIN" --seed "$ROLLOUT_LAYOUT_ID" \
    --max-decisions "${MAX_DECISIONS:-0}" "${QUERY_ARGS[@]}" >"$ROLLOUT_ARCHIVE/logs/controller.log" 2>&1 & CONTROLLER_PID=$!
wait "$CONTROLLER_PID"; status=$?; CONTROLLER_PID=''; exit "$status"

#!/usr/bin/env bash
# Start/stop one RoboDojo simulator on the 4090 host.  All state is on the
# shared /mnt/cfs mount so the A100 launcher can observe readiness and logs.
set -euo pipefail
action=${1:?start, status or stop}; shift
alive() {
    [[ -f "$1" && -f "$1.start" ]] || return 1
    local pid born
    pid=$(<"$1"); born=$(<"$1.start")
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/stat" ]] || return 1
    [[ "$(awk '{print $22}' "/proc/$pid/stat")" == "$born" ]] || return 1
    [[ "$(awk '{print $3}' "/proc/$pid/stat")" != Z ]] || return 1
    kill -0 "$pid" 2>/dev/null
}
case "$action" in
start)
    task=${1:?task}; eval_seed=${2:?eval_seed}; port=${3:?port}; output=${4:?output}
    manifest=${5:?manifest}; case_file=${6:?case_file}; root=${7:?RoboDojo source}
    python=${8:?RoboDojo python}; code_root=${9:?code root}; pid_file=${10:?pid file}; log_file=${11:?log file}
    gpu=${12:-0}; env_script=${13:-}; verify_skip=${14:-0}; scene_query=${15:-0}
    motion_tools=${16:-0}
    collection_steps=${17:-0}
    pi05_collision_shadow=${18:-0}
    [[ "$pi05_collision_shadow" == 0 || "$pi05_collision_shadow" == 1 ]] || { echo 'Invalid pi05 collision shadow flag' >&2; exit 2; }
    [[ "$collection_steps" =~ ^[0-9]+$ ]] || { echo 'Invalid collection step limit' >&2; exit 2; }
    [[ "$motion_tools" == 0 || "$motion_tools" == 1 ]] || { echo "Invalid motion tools flag" >&2; exit 2; }
    [[ "$scene_query" == 0 || "$scene_query" == 1 ]] || { echo "Invalid scene query flag" >&2; exit 2; }
    mkdir -p "$(dirname "$pid_file")" "$output"
    [[ ! -e "$pid_file" && ! -e "$pid_file.start" ]] || { echo 'Existing simulator owner record' >&2; exit 2; }
    "$python" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1])))' "$port"
    [[ -z "$env_script" ]] || source "$env_script"
    # Explicit campaign settings cross SSH; absent settings preserve legacy
    # remote/environment defaults. Apply after env_script to keep frozen settings.
    gripper_contact=${19:-}; pi_force_hold=${20:-}; pi_force_hold_config=${21:-}
    gripper_hold_config=${22:-}
    for flag in "$gripper_contact" "$pi_force_hold"; do
        [[ -z "$flag" || "$flag" == 0 || "$flag" == 1 ]] || { echo 'Invalid gripper feature flag' >&2; exit 2; }
    done
    [[ -z "$gripper_contact" ]] || export ROBODOJO_GRIPPER_CONTACT_ENABLED="$gripper_contact"
    [[ -z "$pi_force_hold" ]] || export ROBODOJO_PI_FORCE_HOLD_ENABLED="$pi_force_hold"
    [[ -z "$pi_force_hold_config" ]] || export ROBODOJO_PI_FORCE_HOLD_CONFIG="$pi_force_hold_config"
    [[ -z "$gripper_hold_config" ]] || export ROBODOJO_GRIPPER_HOLD_CONFIG="$gripper_hold_config"
    # This launcher intentionally uses the deployment-overlay mode: panel and
    # selected layout hashes remain checked, while native source checkout
    # hashes are audited separately on the two hosts.
    export ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY="$verify_skip"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$code_root:$root:$root/XPolicyLab:$root/third_party/curobo"
    "$python" -c 'import json,sys; from pathlib import Path; from hybrid_rollout.robodojo.deployment import native_identity; expected=json.loads((Path(sys.argv[1]).parent/"resume_identity.json").read_text()); assert not expected or native_identity(sys.argv[2]) == expected["native_4090"], "4090 source changed during campaign"' "$output" "$root"
    # CPU-only preflight: dependency/asset failures must precede simulator launch.
    "$python" -m hybrid_rollout.robodojo.robodojo_server.cooked_collision \
        --asset-root "$root/Assets/Robots/x5"
    args=(--task "$task" --eval-seed "$eval_seed" --port "$port" --output "$output"
          --eval-manifest "$manifest" --case-file "$case_file")
    [[ "$scene_query" == 0 ]] || args+=(--enable-scene-query)
    [[ "$motion_tools" == 0 ]] || args+=(--enable-motion-tools)
    [[ "$pi05_collision_shadow" == 0 ]] || args+=(--pi05-collision-shadow)
    [[ "$collection_steps" == 0 ]] || args+=(--collection-step-limit "$collection_steps")
    if [[ "$motion_tools" == 1 ]]; then
        "$python" -c 'from hybrid_rollout.robodojo.robodojo_server.joint_motion import VERSION, Limits; print("Planner:", VERSION)'
    fi
    kit='--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera'
    kit+=' --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true'
    kit+=' --/app/hydraEngine/waitIdle=true'
    mkdir -p "$output/native_runtime"
    cd "$output/native_runtime"
    nohup setsid "$python" -X faulthandler -m hybrid_rollout.robodojo.robodojo_server.server \
        "${args[@]}" --kit_args="$kit" >"$log_file" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    awk '{print $22}' "/proc/$pid/stat" > "$pid_file.start"
    ;;
status)
    alive "${1:?pid file}"
    ;;
stop)
    pid_file=${1:?pid file}; port=${2:?port}
    if alive "$pid_file"; then
        pid=$(<"$pid_file")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
        for _ in {1..30}; do alive "$pid_file" || break; sleep 1; done
        if alive "$pid_file"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"
        fi
        for _ in {1..5}; do alive "$pid_file" || break; sleep 1; done
        alive "$pid_file" && { echo 'Simulator still alive' >&2; exit 3; }
    fi
    # Preserve owner records; remote port release must be explicitly confirmed.
    python3 -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",int(sys.argv[1])))' "$port"
    ;;
*) echo "usage: $0 start|stop ..." >&2; exit 2;;
esac

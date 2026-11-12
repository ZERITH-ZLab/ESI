#!/usr/bin/env bash
# Sourced by the episode owner. One model load per local host, across experiments
# and GPUs: CPU RAM is shared. Never unlink the lock file (waiters use its inode).
PI05_LOAD_FD=''
pi05_load_acquire() {
    local started=$SECONDS timeout="${ROLLOUT_PI05_LOAD_WAIT_TIMEOUT:-7200}" rc
    local path="${ROLLOUT_PI05_LOAD_LOCK:-/tmp/robodojo-pi05-load.lock}"
    [[ "$timeout" =~ ^[1-9][0-9]*$ ]] || {
        echo 'ROLLOUT_PI05_LOAD_WAIT_TIMEOUT must be positive integer seconds' >&2
        return 3
    }
    exec {PI05_LOAD_FD}>>"$path"
    printf '{"event":"pi05_load_waiting","pid":%s}\n' "$$"
    # Poll instead of a blocking flock so INT/TERM reach the owner's cleanup
    # promptly. Waiting has its own timeout, not the model startup deadline.
    while true; do
        if flock -n "$PI05_LOAD_FD"; then break; else rc=$?; fi
        if (( rc != 1 )); then
            echo "pi05 load lock failed: flock exit $rc" >&2
            return 3
        fi
        if (( SECONDS-started >= timeout )); then
            echo 'pi05 load slot wait timeout (model has not started)' >&2
            return 3
        fi
        sleep 1
    done
    printf '{"event":"pi05_load_acquired","pid":%s,"wait_seconds":%s}\n' "$$" "$((SECONDS-started))"
}

pi05_load_release() {
    [[ -n "${PI05_LOAD_FD:-}" ]] || return 0
    # Explicit unlock is essential: the model inherits the descriptor, so just
    # closing the parent's fd would serialize entire episodes, not model loads.
    # On abnormal owner death the inherited fd protects a still-loading child.
    flock -u "$PI05_LOAD_FD"
    exec {PI05_LOAD_FD}>&-
    PI05_LOAD_FD=''
    printf '{"event":"pi05_load_released","pid":%s}\n' "$$"
}

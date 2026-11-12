#!/usr/bin/env bash
# Ten-task pilot; re-running resumes completed/incomplete task campaigns safely.
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$ENTRY_ROOT/../.." && pwd)"
export ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-/opt/RoboDojo}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/envs/RoboDojo/bin/python}"
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
exec bash "$ENTRY_ROOT/launch_dual.sh" --collection-suite "$@"

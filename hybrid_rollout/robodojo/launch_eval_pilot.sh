#!/usr/bin/env bash
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$ENTRY_ROOT/../.." && pwd)"
export ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-/opt/RoboDojo}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/envs/RoboDojo/bin/python}"
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-gpt-6-astra}"
export ROLLOUT_REASONING_EFFORT="${ROLLOUT_REASONING_EFFORT:-xhigh}"
export ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT="${ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT:-96000}"
exec "$ROBODOJO_PYTHON" -u -m hybrid_rollout.robodojo.eval_pilot "$@"

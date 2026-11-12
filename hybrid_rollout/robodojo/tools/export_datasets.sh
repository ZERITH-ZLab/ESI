#!/usr/bin/env bash
# Local offline export only. Pass --index, --output and --action-label explicitly.
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPORT_PYTHON="${EXPORT_PYTHON:-/opt/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}"
EXPORT_DEPS="${EXPORT_DEPS:-/opt/esi-export-deps}"
export PYTHONPATH="$EXPORT_DEPS:$ENTRY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/robodojo-export-datasets-cache}"
exec "$EXPORT_PYTHON" -u -m hybrid_rollout.robodojo.tools.export_datasets "$@"

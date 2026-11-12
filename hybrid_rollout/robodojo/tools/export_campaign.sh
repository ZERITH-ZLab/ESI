#!/usr/bin/env bash
# Frozen input -> full two-episode pilot -> parallel command-label export shards.
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPORT_PYTHON="${EXPORT_PYTHON:-/opt/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python}"
EXPORT_DEPS="${EXPORT_DEPS:-/opt/esi-export-deps}"
export PYTHONPATH="$EXPORT_DEPS:$ENTRY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec "$EXPORT_PYTHON" -u -m hybrid_rollout.robodojo.tools.export_campaign "$@"

#!/usr/bin/env bash
# One process owns all eight GPUs. Do not wrap this in torchrun/mpirun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Without an explicit native config, use the release deployment/recipe interface.
if [[ -z "${TRAIN_CONFIG:-}" ]]; then
  exec bash scripts/train.sh "$@"
fi
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON for a native training config, or use scripts/train.sh}"
# Respect the scheduler's visible devices; device count is validated by JAX.
export JAX_PLATFORMS=cuda
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}}"
unset XLA_PYTHON_CLIENT_MEM_FRACTION
exec "${TRAIN_PYTHON}" -u \
  -m data_processing.training run \
  --config "${TRAIN_CONFIG}" "$@"

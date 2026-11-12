#!/usr/bin/env bash
# Platform starts this ONCE PER NODE. No torchrun; JAX owns all local GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export JAX_PROCESS_COUNT="${JAX_PROCESS_COUNT:-${WORLD_SIZE:?Set WORLD_SIZE to number of nodes}}"
export JAX_PROCESS_INDEX="${JAX_PROCESS_INDEX:-${RANK:?Set RANK to node index}}"
if [[ -z "${TRAIN_CONFIG:-}" ]]; then
  exec bash scripts/train.sh "$@"
fi
# Native-config compatibility: topology/batches come from TRAIN_CONFIG or explicit
# TRAIN_* overrides. No assumption of eight devices per node.
# Workers are per node, so two nodes use 32 in total by default.
# Use the native config worker count unless explicitly overridden.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
# Proxies must not intercept internal JAX coordinator/NCCL connections.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy
exec bash data_processing/training/launch_8gpu.sh "$@"

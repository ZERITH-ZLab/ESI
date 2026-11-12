#!/usr/bin/env bash
# Compatibility name; deployment and experiment identity come from release configs.
set -euo pipefail
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ENTRY_ROOT/../../scripts/eval.sh" "$@"

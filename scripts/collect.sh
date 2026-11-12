#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# --config later in the argument list overrides this recommended recipe.
exec "${ESI_PYTHON:-python3}" -m release_tools --config configs/collection/bootstrap.json "$@"

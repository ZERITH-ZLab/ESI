#!/usr/bin/env bash
# A100: bash hybrid_rollout/robodojo/launch_collection.sh [--dry-run|resume|resume-upgrade]
set -euo pipefail

# 常改配置：新实验换名字；中断后用同名 + resume，不修改策略源码。
export ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:-arrange_largest_number_collect20x5_abandon_v1}"
# 每个 case 的采集轮数；改变轮数请使用新的实验名和独立 manifest。
export ROLLOUT_COLLECTION_REPEATS="${ROLLOUT_COLLECTION_REPEATS:-5}"
if [[ ! "$ROLLOUT_COLLECTION_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ROLLOUT_COLLECTION_REPEATS must be a positive integer' >&2; exit 2
fi
export ROLLOUT_ENABLE_TEACHER_ABANDON="${ROLLOUT_ENABLE_TEACHER_ABANDON:-1}"
# 曾连续纠错 >=60 步才总结；注入最近4份+其余纠错占比最高4份。变更请用新实验名。
export ROLLOUT_ENABLE_TASK_MEMORY="${ROLLOUT_ENABLE_TASK_MEMORY:-1}"
# 该文件中的 profiles 顺序应为 codex_a_3、codex_a_4；运行时允许热更新。
export ROLLOUT_ACCOUNT_POOL_FILE="${ROLLOUT_ACCOUNT_POOL_FILE:-/root/.config/gpt-as-policy/account_pool.json}"
# 机器和端口仍由 launch_dual.sh 管理，环境变量可覆盖。
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
export ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-/opt/RoboDojo}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/envs/RoboDojo/bin/python}"

# 完整20-case面板不变；显式 CASE_IDS 可选子集，默认计划20 × repeats。
export ROBODOJO_TASK=arrange_largest_number ROLLOUT_COUNT=20
export ROLLOUT_CONTROLLER_ROUTE=auto_infer_v1 ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt
export ROLLOUT_ENABLE_SCENE_QUERY=1 ROLLOUT_ENABLE_MOTION_TOOLS=1
export ROLLOUT_MODEL=gpt-6-astra ROLLOUT_REASONING_EFFORT=xhigh
export ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT=96000 ROLLOUT_COLLECTION_STEP_LIMIT=1500
export ROLLOUT_MAX_SECONDS=10800 MAX_DECISIONS=0 CODEX_MAX_TOTAL_TOKENS=0
export ROLLOUT_ATTEMPT=auto ROLLOUT_RUN_MODE=fresh
export ROLLOUT_ALLOW_SOURCE_UPGRADE=0
unset ROLLOUT_CASE_ID ROLLOUT_COLLECTION_REPEAT_ID ROLLOUT_COLLECTION_JOB_INDEX
case "${1:-}" in
    resume) export ROLLOUT_RUN_MODE=resume; shift ;;
    resume-upgrade) export ROLLOUT_RUN_MODE=resume ROLLOUT_ALLOW_SOURCE_UPGRADE=1; shift ;;
    --dry-run|'') ;;
    *) echo 'Usage: launch_collection.sh [--dry-run|resume|resume-upgrade]' >&2; exit 2 ;;
esac
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$ENTRY_ROOT/../.." && pwd)"
export ROLLOUT_EVAL_MANIFEST="$ROLLOUT_SHARED_ROOT/collection_panels/arrange_largest_number20_g0_v1.json"
if [[ "$ROLLOUT_COLLECTION_REPEATS" != 5 ]]; then
    export ROLLOUT_EVAL_MANIFEST="$ROLLOUT_SHARED_ROOT/collection_panels/arrange_largest_number20_g0_r${ROLLOUT_COLLECTION_REPEATS}_v1.json"
fi
if [[ "${1:-}" == --dry-run ]]; then
    "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.collection \
        --source "$ROBODOJO_SOURCE" --output "$ROLLOUT_EVAL_MANIFEST" --repeats "$ROLLOUT_COLLECTION_REPEATS" --dry-run
    "$ROBODOJO_PYTHON" -c 'import os; from hybrid_rollout.robodojo.account_pool import read_config; print("Account pool:", read_config(os.environ["ROLLOUT_ACCOUNT_POOL_FILE"])["profiles"])'
    # 新 manifest 尚未写入时只核对本地场景；不创建实验、不请求模型。
    if [[ ! -f "$ROLLOUT_EVAL_MANIFEST" ]]; then exit 0; fi
else
    "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.collection \
        --source "$ROBODOJO_SOURCE" --output "$ROLLOUT_EVAL_MANIFEST" --repeats "$ROLLOUT_COLLECTION_REPEATS"
fi
exec bash "$ENTRY_ROOT/launch_dual.sh" "$@"

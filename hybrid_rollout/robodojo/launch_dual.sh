#!/usr/bin/env bash
# 控制节点执行：bash hybrid_rollout/robodojo/launch_dual.sh [--dry-run]
# 建议在 tmux 内启动。关闭笔记本不会停止服务器上的 tmux。
# Deployment and usage: docs/CONFIGURATION.md. Environment variables override defaults.
set -euo pipefail

# ================= 经常修改：实验选择 =================
# 模型/思考强度：默认保持原版；同一实验固定，改变时使用新 EXPERIMENT_ID。
# 例如 gpt-6-astra + medium，或 gpt-6-sol + high/xhigh。
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-gpt-6-astra}"
export ROLLOUT_REASONING_EFFORT="${ROLLOUT_REASONING_EFFORT:-xhigh}"
# baseline 保留原路线；auto_infer_v1 为自动 infer 新路线。换路线用新实验名。
export ROLLOUT_CONTROLLER_ROUTE="${ROLLOUT_CONTROLLER_ROUTE:-baseline}"
# 仅新路线生效：自动压缩触发阈值，不修改模型上下文容量；改值用新实验名。
export ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT="${ROLLOUT_AUTO_COMPACT_TOKEN_LIMIT:-96000}"
# 跨回合文字经验：auto_infer_v1 / subtask_feedforward_v1 可用，启用/关闭都应换实验名。
export ROLLOUT_ENABLE_TASK_MEMORY="${ROLLOUT_ENABLE_TASK_MEMORY:-0}"
export ROLLOUT_ENABLE_TEACHER_ABANDON="${ROLLOUT_ENABLE_TEACHER_ABANDON:-0}"
case "$ROLLOUT_CONTROLLER_ROUTE" in
    baseline|auto_infer_v1|subtask_feedforward_v1) ;;
    *) echo 'Unknown ROLLOUT_CONTROLLER_ROUTE' >&2; exit 2 ;;
esac
export ROBODOJO_TASK="${ROBODOJO_TASK:-arrange_largest_number}"
# 换任务：只改上一行，例如 arrange_largest_number；replica 从 manifest 自动推导。
# 特权查询允许两种混合控制路线；strict eval 仍禁止，启用时用新实验名。
export ROLLOUT_ENABLE_SCENE_QUERY="${ROLLOUT_ENABLE_SCENE_QUERY:-0}"
# 新路线可独立启用 link6 稀疏 move/grasp；验证可关闭 query 但保留动作工具。
# 连续 S 曲线长运动使用已有 NumPy/SciPy；使用新实验名，详见 skill_auto_infer/RECOVERY_TOOLS.md。
export ROLLOUT_ENABLE_MOTION_TOOLS="${ROLLOUT_ENABLE_MOTION_TOOLS:-0}"
export ROLLOUT_PI05_COLLISION_SHADOW="${ROLLOUT_PI05_COLLISION_SHADOW:-0}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-5}"
# 留空：按 manifest 顺序选前 COUNT 个；非空：只运行列出的 case，忽略 COUNT。
# 例如 fold_clothes__standard__g0__l1
# 多个用逗号：fold_clothes__standard__g0__l1,fold_clothes__random__g0__l0
export ROLLOUT_CASE_IDS="${ROLLOUT_CASE_IDS:-}"
export ROLLOUT_EXPERIMENT_ID="${ROLLOUT_EXPERIMENT_ID:-arrange_largest_number}"
# 已存在的实验不会自动重跑：续跑显式设 RUN_MODE=resume，新实验换名字。
# fresh：新实验，实验/replica 已存在则拒绝；resume：跳过同配置已完成 case，
# 中断的 episode 从头运行（不是从中间物理状态恢复）；rerun：所选 case 全重测。
# 换权重/策略代码/模型设置时，用新 EXPERIMENT_ID + fresh，不混入旧统计。
export ROLLOUT_RUN_MODE="${ROLLOUT_RUN_MODE:-fresh}"
# auto 自动选择新 attempt，永不覆盖旧文件。手动指定必须是未使用的编号。
export ROLLOUT_ATTEMPT="${ROLLOUT_ATTEMPT:-auto}"
# 换号只改这里，先用 profiles login 独立登录；不复制 auth.json。
export ROLLOUT_AUTH_PROFILE="${ROLLOUT_AUTH_PROFILE:-codex_a_2}"
# 可选账号池 JSON 路径；留空保持单账号。启用后忽略上面的单账号选择。
# 每次分配前重读列表；额度耗尽换号从头重试当前 case，全部不可用则醒目提示并等待。
# 示例配置/使用方法：ACCOUNT_POOL.md。账号仍需提前 profiles login。
export ROLLOUT_ACCOUNT_POOL_FILE="${ROLLOUT_ACCOUNT_POOL_FILE:-}"
# 单个 episode 最长秒数；0 不限。步数上限由原生任务决定。
export ROLLOUT_MAX_SECONDS="${ROLLOUT_MAX_SECONDS:-10800}"
export MAX_DECISIONS="${MAX_DECISIONS:-0}"

# ================= 通常不改：A100 / 4090 路径 =================
ENTRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODE_ROOT="$(cd "$ENTRY_ROOT/../.." && pwd)"
export ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-/opt/RoboDojo}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/envs/RoboDojo/bin/python}"
export OPENPI_SOURCE="${OPENPI_SOURCE:-$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-$OPENPI_SOURCE/.venv/bin/python}"
export CHECKPOINT="${CHECKPOINT:-/mnt/models/robodojo_pi05/59999}"
export CODEX_BIN="${CODEX_BIN:-/usr/local/bin/codex}"
export ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-/mnt/esi}"
# 认证目录和实验目录独立；不是同一个 shared-root。
export ROLLOUT_AUTH_ROOT="${ROLLOUT_AUTH_ROOT:-/mnt/rollout/robodojo_mixed_control}"
export ROLLOUT_CODEX_HOME_DIR="$ROLLOUT_AUTH_ROOT/private/auth_profiles/$ROLLOUT_AUTH_PROFILE/codex_home"
export ROLLOUT_SIM_HOST="${ROLLOUT_SIM_HOST:-simulator.example.invalid}"
# 每个槽位使用部署配置中的 SSH 与 GPU 设置。
# 后续并行时每个 worker 独立设置机器、GPU 和服务端口，不复制脚本。
export ROLLOUT_SIM_SSH_USER="${ROLLOUT_SIM_SSH_USER:-root}"
export ROLLOUT_SIM_SSH_PORT="${ROLLOUT_SIM_SSH_PORT:-22}"
export ROLLOUT_SIM_SSH_KEY="${ROLLOUT_SIM_SSH_KEY:-/path/to/simulator_ssh_key}"
export ROLLOUT_SIM_ROBODOJO_SOURCE="${ROLLOUT_SIM_ROBODOJO_SOURCE:-/opt/RoboDojo}"
export ROLLOUT_SIM_PYTHON="${ROLLOUT_SIM_PYTHON:-/opt/envs/RoboDojo/bin/python}"
# 可选：模拟器图形库环境脚本的绝对路径。留空使用机器已配置的运行环境。
export ROLLOUT_SIM_ENV_SCRIPT="${ROLLOUT_SIM_ENV_SCRIPT:-}"
# GPU 是各自机器的 CUDA 编号，不是 vulkaninfo 的枚举编号。
export POLICY_GPU="${POLICY_GPU:-0}" SIM_GPU="${SIM_GPU:-0}"
# POLICY_PORT：策略服务；SIM_PORT：本机 SSH 转发；REMOTE_PORT：远端模拟器。
# 后续并行需要独立端口、case 分片和账号资源；当前仍是一个串行 worker。
export POLICY_PORT="${POLICY_PORT:-18830}" SIM_PORT="${SIM_PORT:-19113}"
export ROLLOUT_SIM_REMOTE_PORT="${ROLLOUT_SIM_REMOTE_PORT:-19113}"
export ROLLOUT_STARTUP_TIMEOUT="${ROLLOUT_STARTUP_TIMEOUT:-900}"
# 同机所有 worker 串行加载 π05，加载并校验后释放锁；推理仍并行。
# 不要按 worker/GPU 改锁路径，CPU 内存由同机所有进程共享。
export ROLLOUT_PI05_LOAD_LOCK="${ROLLOUT_PI05_LOAD_LOCK:-/tmp/robodojo-pi05-load.lock}"
export ROLLOUT_PI05_LOAD_WAIT_TIMEOUT="${ROLLOUT_PI05_LOAD_WAIT_TIMEOUT:-7200}"
export ROLLOUT_EVAL_MANIFEST="${ROLLOUT_EVAL_MANIFEST:-$ENTRY_ROOT/eval_panels/robodojo_panel60_v1.json}"
export ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY="${ROLLOUT_SKIP_NATIVE_SOURCE_VERIFY:-1}"
export ROLLOUT_EVALUATION_METHOD="${ROLLOUT_EVALUATION_METHOD:-pi05_plus_gpt}"
export ROLLOUT_ALLOW_SEQUENTIAL_BATCH=1
export ROLLOUT_EPISODE_ENTRYPOINT="$ENTRY_ROOT/dual_machine_episode_entrypoint.sh"
# 代理继承当前终端；如需 clashctl on，请先在该 tmux 窗口执行。
# 不读取旧 .env.server，避免两个配置来源互相覆盖。
unset ROLLOUT_CASE_ID ROLLOUT_SEED_BASE ROLLOUT_RERUN_REFERENCE
unset ROLLOUT_CODEX_STATE_DIR OPENAI_API_KEY OPENAI_BASE_URL CODEX_API_KEY
export PYTHONPATH="$CODE_ROOT"
if [[ "${1:-}" == --eval-retry ]]; then
    shift
    exec "$ROBODOJO_PYTHON" -u -m hybrid_rollout.robodojo.eval_retry "$@"
fi
if [[ "${1:-}" == --eval-followup ]]; then
    shift
    exec "$ROBODOJO_PYTHON" -u -m hybrid_rollout.robodojo.eval_followup "$@"
fi
if [[ "${1:-}" == --eval-queue ]]; then
    shift
    exec "$ROBODOJO_PYTHON" -u -m hybrid_rollout.robodojo.eval_queue "$@"
fi
if [[ "${1:-}" == --collection-suite ]]; then
    shift
    exec "$ROBODOJO_PYTHON" -u -m hybrid_rollout.robodojo.collection_suite "$@"
fi
exec "$ROBODOJO_PYTHON" -m hybrid_rollout.robodojo.dual_campaign "$@"

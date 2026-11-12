# ESI: A Self-Improving Flywheel for LLM–VLA Agents

ESI 将 LLM 规划与 VLA 策略结合，通过采集、子任务标注与训练持续改进系统。

## 复现最终 ESI 评测

默认评测使用最终 ESI checkpoint、GPT 前馈子任务和项目提供的任务指导，采用 benchmark 原生预算与评分。默认评测集为 10 个任务，每任务 20 个 case。

**最终权重下载地址待发布。** 权重不包含在代码仓库中；目录结构与发布状态见 [CHECKPOINTS.md](docs/CHECKPOINTS.md)。

1. 按[安装教程](docs/INSTALLATION.md)安装 RoboDojo、XPolicyLab/OpenPI、Isaac Sim/IsaacLab、cuRobo 和资产。教程包含推荐路径文件树与必要补丁。
2. 将最终权重放到 `/mnt/models/esi-final`，或在部署配置中填写实际路径。
3. 按[评测部署教程](docs/EVALUATION.md)配置设备、路径和账号。模型与策略已有默认配置；主机地址、认证、GPU 和并行容量需要根据自己的设备填写。
4. 在仓库根目录执行：

```bash
# 使用部署配置中的控制工具 Python 环境。
export ESI_PYTHON=/opt/envs/RoboDojo/bin/python
bash scripts/eval.sh --set run.name=esi_final_eval --show-config
bash scripts/eval.sh --set run.name=esi_final_eval
```

入口自动启动模拟器、π05 服务与 GPT 控制器。长任务建议在 tmux 内运行。恢复时使用相同命令和配置，已完成 episode 会保留；配置预览本身不检查 GPU、SSH 或账号是否可用。

## 其他路线

| 路线 | 入口 | 教程 |
| --- | --- | --- |
| 完整后训练飞轮 | `scripts/collect.sh`、`annotate.sh`、`data.sh`、`train.sh` | [分阶段工作流](docs/CONFIGURATION.md) |
| 基准评测 | `scripts/eval.sh --config configs/eval/baseline.json` | [评测部署](docs/EVALUATION.md) |
| 可选自动 prompt 迭代 | `scripts/evolve.sh` | [Prompt 迭代](hybrid_rollout/robodojo/prompt_evolution/README.md) |

自动 prompt 迭代默认关闭。飞轮各阶段可独立运行，不要求用一个命令编排所有轮次。

## 依赖

外部源码、完整资产、账号和 checkpoint 需单独准备，见[依赖说明](docs/DEPENDENCIES.md)。部署后请先按安装教程进行环境检查和小规模试运行。

高级配置见 [CONFIGURATION.md](docs/CONFIGURATION.md)，快照与恢复机制见 [EVAL_SNAPSHOTS.md](docs/EVAL_SNAPSHOTS.md)。部署时使用 `scripts/` 和 `configs/` 中的统一入口。

许可见 [LICENSE](LICENSE) 和[第三方声明](THIRD_PARTY_NOTICES.md)。外部源码、资产和权重遵循各自许可。

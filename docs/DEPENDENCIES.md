# 外部依赖与安装目录

ESI 不包含模拟器、外部策略代码、完整资产和模型权重。安装这些依赖后，在 `configs/deployment.local.json` 中填写实际路径；安装根目录可自定义。

## 所需组件

| 组件 | 用途与接口要求 |
| --- | --- |
| RoboDojo | 原生任务、布局、评分与模拟器接口 |
| XPolicyLab / OpenPI | RoboDojo 适配的 π05 推理及 JAX 训练；需要 `pi05_base_aloha_full_sim_arx-x5_seed_0` 配置 |
| Isaac Sim / IsaacLab | 模拟器运行环境；当前启动接口要求 Isaac Sim 5.1 系列 |
| cuRobo | GPT 运动规划使用的运动学与碰撞相关依赖 |
| RoboDojo Assets | 任务物体、机器人及场景资产 |
| 官方 π05 checkpoint | 首轮采集和基准评测；目录需包含 `params/` 与 `assets/arx_x5_sim/norm_stats.json` |
| Codex CLI | 在线控制、标注与可选 prompt 迭代；使用部署者自己的认证 |
| ffmpeg / ffprobe | 视频处理和可视化 |

## 获取与目录组织

- [RoboDojo 源码](https://github.com/RoboDojo-Benchmark/RoboDojo) 和 [官方文档](https://robodojo-benchmark.com/)：按选定版本的说明安装模拟器环境、任务资产和权重。
- [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)：安装与 RoboDojo 相配套的策略集成代码。
- [OpenPI 上游](https://github.com/Physical-Intelligence/openpi) 供参考；本项目需要 XPolicyLab 中的 RoboDojo 适配配置，不能直接用任意上游分支替代。

推荐目录布局：

```text
<external-root>/RoboDojo/
  XPolicyLab/
    policy/Pi_05/openpi/
  third_party/IsaacLab/
  third_party/curobo/

<storage-root>/Assets/
<storage-root>/checkpoints/
<storage-root>/experiments/
<source-root>/ESI/
```

在 `configs/deployment.local.json` 的 `paths` 中配置：

- `robodojo_source`、`robodojo_python`：本机 RoboDojo 路径及工具运行解释器。
- `openpi_source`、`openpi_python`：适配版 OpenPI 路径及其独立 Python 环境。
- `sim_robodojo_source`、`sim_python`：模拟器节点上的安装路径和解释器。
- `official_checkpoint`、`output_root`：权重与实验输出目录，建议使用大容量挂载盘。
- `codex_bin`、`auth_root`、`account_pool_file`、`sim_ssh_key`：外部认证与连接配置，真实凭据不要提交到 Git。

完整配置及各阶段命令见 [CONFIGURATION.md](CONFIGURATION.md)。多机执行需要 SSH、空闲端口和可用 GPU；当前远程执行流程要求相关节点能通过相同路径读写共享实验存储。

## 版本与兼容性

参考源码版本：

| 项目 | 参考 commit |
| --- | --- |
| RoboDojo | `08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d` |
| XPolicyLab | `bb9a0b5f5136a74503b679af830bfd0a3a837d5c` |
| IsaacLab（RoboDojo fork） | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| cuRobo（RoboDojo fork） | `895c6517243f8cb091c73c018c8167192d39599a` |

OpenPI 位于 XPolicyLab 的策略目录内，XPolicyLab commit 不是 OpenPI 上游 commit。这些版本号不是完整的可复现环境锁；Python 依赖、Isaac 运行时、cuRobo 与资产兼容性仍需按部署环境验证。请在自己的设备上完成安装检查后再启动完整任务。

cuRobo 需要上述 fork 的 v2 接口（`BatchMotionPlanner`、`InverseKinematics` 等），不能直接用 RoboDojo gitlink 的 `d17b54ce…` 或任意最新版替代。对应补丁、安装命令及验证步骤见 [INSTALLATION.md](INSTALLATION.md)。

碰撞几何与源资产需要匹配。安装路径可以自定义，但更换机器人或碰撞参数需要重新生成并验证派生几何。按安装教程更新资产路径并运行预检；不要关闭校验绕过不匹配。

## Python 环境与运行检查

模拟器与 OpenPI 使用独立环境，不要将所有依赖混装并自动升级：

- `requirements-tools.txt`：轻量工具依赖，不包含完整模拟器或训练环境。
- `data_processing/requirements.txt`：数据处理依赖。
- `hybrid_rollout/robodojo/requirements-motion.txt`：运动几何相关依赖。
- `data_processing/policy_review/requirements.txt`：策略可视化依赖。

先用入口的 `--show-config` 检查配置，再以少量 case 和小 batch 验证真实服务。配置预览不代表 GPU、外部安装、账号或网络已可用。代理按网络条件配置，不要求使用特定代理软件。

外部项目、权重和资产遵循各自许可；本项目许可不替代其授权。

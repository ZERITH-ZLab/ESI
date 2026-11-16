# 配置与分阶段启动

各阶段独立启动，配置会随输出保存。所有示例从仓库根目录执行；`/mnt/esi` 是示例输出盘，请换成本机挂载路径。

`paths.output_root` 可以是任意可创建、可读写的绝对目录，例如 `/data/esi` 或 `/home/user/esi-output`，不能位于源码仓库内或直接使用文件系统根目录。多机部署时，各节点必须通过相同绝对路径访问同一份共享存储；启动前会验证两端的文件读写可见性。

最终版评测的最短部署路径见 [EVALUATION.md](EVALUATION.md)。下面是完整飞轮及高级配置参考。

## 配置放在哪里

| 内容 | 配置位置 | 用途 |
| --- | --- | --- |
| 环境路径、账号文件、模拟器/GPU 槽位 | `configs/deployment.local.json` | 调用现有 SSH 服务与账号池；认证文件仍在仓库外 |
| 在线 GPT 与标注模型 | `configs/llm.json` | 在线 astra/xhigh，标注 sol/medium；名称可编辑，运行前确认账号支持 |
| 任务与评测 case | `configs/tasks.json`、`configs/eval/cases.json` | 10 任务，每任务 20 case；有 random 的任务 standard/random 各 10 |
| 首轮 / 后续采集 | `configs/collection/bootstrap.json`、`feedforward.json` | 预算倍率 10/7、自动保持和归零收紧；首轮无前馈、后续有前馈 |
| RoboDojo 执行适配 | `configs/environments/robodojo.json` | 自动保持/收紧、GPT 闭合等待参数、最终 eval 下采样参数；仅适用于当前 RoboDojo/X5 接口 |
| 离线 / 在线标注 | `configs/annotation/offline.json`、`online.json` | 轨迹分段 / 在线指令结果评估 |
| 数据准备 | `configs/data.json` | command 动作导出、有效范围、索引、跨轮合并、归一化 |
| 概率采样 | `configs/sampling.json` | 任务等权、GPT 起点权重 4、π05 起点 1、成功 case 权重 2、其他 1 |
| 训练 | `configs/training/round*.json` | subtask 主配方；超参数集中在 config.training / optimizer / checkpoint |
| 最终 / 基准评测 | `configs/eval/final.json`、`baseline.json` | 评测任务与恢复设置 |
| 自动 prompt 迭代 | `configs/prompt_evolution.json` | 默认关闭，仅持续主分析 + 独立持续终审 |

JSON 支持 `extends`、`{"$file":"相对路径"}` 和 `${字段.引用}`。相对 include 路径以所在配置文件为基准；普通数据/安装路径建议填绝对路径。`--set KEY=VALUE` 只允许覆盖已存在的键，JSON 布尔值用 `true/false`；数组要用 shell 单引号保护。未知键报错，避免拼写错误静默生效。

复制部署模板一次：

```bash
cp configs/deployment.example.json configs/deployment.local.json
# 编辑路径、账号、SSH、槽位；模板中的 example.invalid 主机不能运行。
export ESI_PYTHON=/path/to/robodojo/environment/bin/python
```

`paths.sim_ssh_user` 配置模拟器 SSH 用户，默认 root 仅为兼容；每个槽位可用 `ROLLOUT_SIM_SSH_USER` 覆盖。最终 eval 默认读取 `paths.final_checkpoint`（`/mnt/models/esi-final`），基准仍读取 `paths.official_checkpoint`。

`ESI_PYTHON` 用于运行配置入口；真正的模拟器/标注解释器、OpenPI 解释器来自 deployment.paths。账号登录与账号池格式沿用 [ACCOUNT_POOL.md](../hybrid_rollout/robodojo/ACCOUNT_POOL.md)。模板的 codex_a 只是槽位名称，必须先登录。prompt 迭代复用 `deployment.annotation` 的账号设置。

`configs/deployments/` 提供单槽位、小批量训练、双节点 16 卡模板。这些文件继承 **example**，不会自动合并你的 local 配置。可复制所需字段到 local，或新建 local 文件，用 `extends` 指向你已有的 local 配置再覆盖硬件字段。

## 预览、输出与恢复

所有入口支持 `--deployment 文件`、`--config 配方`、重复的 `--set`、`--show-config`。
`--show-config` 不写文件、不启动模型、SSH、GPU 或账号请求，只展示解析后的配置及底层命令。它不能证明外部安装和账号可用。

必须提供新的 `run.name`。启动后配置保存于：

```
<output_root>/release_runs/<run.name>/<stage>/[action/]
  resolved_config.json
  stage_config.json
  annotation_accounts.json   # 需要时生成；只含账号路径/槽位，不含凭据
```

同名同阶段配置不可被覆盖；变化会报错。更换策略、路径、采样规则或模型请使用新名字。采集/eval 的断点恢复沿用原调度器，保留已完成 case，未完成 episode 重新开始。训练恢复显式在命令末尾加 `-- --resume`。数据阶段会检查已存在产物身份。

运行会记录代码、权重和环境身份，恢复时进行校验。详见[快照与恢复](EVAL_SNAPSHOTS.md)。

入口清理继承的 rollout/simulator/policy 环境变量，避免其他实验的设置干扰当前配置。自定义底层 rollout 参数应写在配方 `environment` 中。代理环境继承终端；多节点训练会移除代理，避免拦截节点间通信。

## 1. 采集

```bash
bash scripts/collect.sh --set run.name=round1_collection --show-config
bash scripts/collect.sh --set run.name=round1_collection

bash scripts/collect.sh --config configs/collection/feedforward.json \
  --set run.name=round2_collection --set run.round=2 \
  --set run.checkpoint=/mnt/esi/training/round1_train/checkpoints/6000
```

`run.checkpoint` 指向同时含 `params/` 和 `assets/` 的推理 checkpoint 根目录。

每个 `deployment.collection.slots` 指定本机 policy GPU/端口、模拟器主机/SSH 端口/GPU/服务端口；可有任意正数槽位。workers 不得超过槽位数。GPU 可共享，但容量由部署者负责；模板的 8 个 policy 实例共享 GPU 0 不表示任意显卡都能容纳。多路仍需要账号池及足够并发额度。模拟器主机之间必须安装相同的原生任务代码和资产。

### RoboDojo 执行适配配置

`configs/environments/robodojo.json` 集中保存当前环境的执行参数，由采集和评测配方显式继承。这些不是 ESI 飞轮的通用要求：自动保持/收紧依赖 RoboDojo/X5 的夹爪映射（0 表示闭合）、PhysX 接触与原生插值；GPT 闭合等待依赖实际夹爪位置反馈；下采样阈值依赖动作单位和控制步频率。更换环境应提供对应适配，不直接继承这些参数。

| 设置 | 采集默认 | 最终 eval | 基准 eval |
| --- | --- | --- | --- |
| `robodojo.pi_force_hold.enabled` | `true` | `false`，路线固定关闭 | `false`，路线固定关闭 |
| `robodojo.pi_force_hold.config.tighten_enabled` | `true`，`tighten_mode=zero` | 随总开关关闭，不生效 | 随总开关关闭，不生效 |
| `robodojo.gpt_gripper_hold` | GPT 显式闭合等待参数 | 同左 | 同左 |
| `robodojo.eval_downsampling` | 采集不使用 | 引用该参数组 | 固定 `off` |

采集可选择以下两个关闭层级（也适用于 `feedforward.json`）：

```bash
# 保留自动抓取保持，不修改模型的夹爪目标
bash scripts/collect.sh --set run.name=hold_only \
  --set robodojo.pi_force_hold.config.tighten_enabled=false

# 关闭整套自动保持和收紧；GPT 显式闭合等待仍保留
bash scripts/collect.sh --set run.name=no_auto_intervention \
  --set robodojo.pi_force_hold.enabled=false
```

GPT 闭合等待与自动干预独立配置。最终 eval 的下采样可用 `--set robodojo.eval_downsampling.mode=off` 关闭。更换环境时应重新验证这些适配参数。

运行中调整并发继续使用调度器打印的 worker control 文件。只允许 0 到已配置槽位总数；0 为停止派发并等待在途 episode 保存退出；调度器保持暂停，可重新扩容。确认 active_workers=0 后可结束调度器。不要为在线调整修改已冻结的配方。

## 2. 标注

先 prepare，再 run；第二轮起将配方改为 online，其他操作相同：

```bash
bash scripts/annotate.sh --set run.name=round1_labels \
  --set 'run.inputs=["/mnt/esi/results/round1_collection__task_a","/mnt/esi/results/round1_collection__task_b"]'
bash scripts/annotate.sh --set run.name=round1_labels --set run.action=run
bash scripts/annotate.sh --set run.name=round1_labels --set run.action=validate

bash scripts/annotate.sh --config configs/annotation/online.json \
  --set run.name=round2_labels --set 'run.inputs=["/path/to/completed/task/results"]'
bash scripts/annotate.sh --config configs/annotation/online.json \
  --set run.name=round2_labels --set run.action=run
```

输入须列出实际任务结果目录；示例的 task_a/task_b 需替换。默认 batch 为 `<output_root>/annotations/<name>`。运行失败后重跑相同 run 命令，已校验的标注会跳过。底层额外选项放在 `--` 后，例如 `-- --limit 2`。不要用截断样本的测试结果直接替代完整训练数据准备。

## 3. 导出、索引、归一化

先从标注 batch 的 `latest_run.json` 确认成功运行的实际 `runs/<id>` 目录。不同数据阶段建议使用不同 run.name，尤其合并和归一化会产生独立产物。

```bash
bash scripts/data.sh --set run.name=round1_export --set run.action=export \
  --set run.annotation_run=/path/to/annotations/runs/verified_run \
  --set run.output=/mnt/esi/exports/round1
bash scripts/data.sh --set run.name=round1_index --set run.action=index \
  --set run.annotation_run=/path/to/annotations/runs/verified_run \
  --set 'run.exports=["/mnt/esi/exports/round1"]' \
  --set run.policy_version=official --set run.collection_round=round1 \
  --set run.output=/mnt/esi/indexes/round1
bash scripts/data.sh --set run.name=round2_merge --set run.action=merge \
  --set 'run.indexes=["/mnt/esi/indexes/round1","/mnt/esi/indexes/round2"]' \
  --set run.output=/mnt/esi/indexes/round1_2
bash scripts/data.sh --set run.name=round2_norm --set run.action=normalize \
  --set run.index=/mnt/esi/indexes/round1_2 \
  --set run.checkpoint=/mnt/esi/training/round1_train/checkpoints/6000 \
  --set run.output=/mnt/esi/norm/round1_2
```

导出使用实际执行的 command 动作。索引根据标注确定监督范围，并保存无效动作的损失掩码；归一化从有效监督数据计算。图像仍引用原始观测路径，训练节点须能访问原数据及导出、索引、归一化文件。

## 4. 训练

```bash
bash scripts/train.sh --config configs/training/round1.json \
  --set run.name=round1_train --set run.index=/mnt/esi/indexes/round1 \
  --set run.norm=/mnt/esi/norm/round1 --show-config

bash scripts/train.sh --config configs/training/round2.json \
  --set run.name=round2_train --set run.index=/mnt/esi/indexes/round1_2 \
  --set run.norm=/mnt/esi/norm/round1_2 \
  --set run.checkpoint=/mnt/esi/training/round1_train/checkpoints/6000
```

删掉 `--show-config` 才启动；先用 `-- --stop-after 2` 做小步检查，再用相同配置加 `-- --resume` 恢复。单卡小批量测试应显式选择小型部署参数，不直接用 8 卡配方启动。

默认：subtask、6000 更新、统一 VLM/动作专家峰值 LR=2e-5、warmup600、末尾1e-6、每1000步保存、最近2份、预分配显存、显存比例0.75。选择普通训练用 `--set run.mode=task`。常用参数可通过 `--set config.optimizer.learning_rate=...`、`--set config.training.num_steps=...` 等更改。已有 `TRAIN_*` 覆盖接口仍有效，优先于 JSON，覆盖后的实际配置也会保存；新部署建议选一种方式，不混用。

| 部署 | 节点 / 总 GPU | 每卡 batch | GA | 全局 micro batch | 全局 batch | 每节点取数进程 |
| --- | --- | --- | --- | --- | --- | --- |
| 默认 | 1 / 8 | 8 | 4 | 64 | 256 | 32 |
| train16 示例 | 2 / 16 | 8 | 2 | 128 | 256 | 16 |

关系为 `global_batch = num_devices × per_device_batch_size × gradient_accumulation_steps`。改变设备数时，请相应调整 GA。FSDP 分组默认8，设备数须能整除分组。每个节点启动**一个** JAX 进程，使用本节点所有分配的 GPU；不要再按每 GPU 启动 torchrun。

两节点平台分别提供节点 `RANK=0/1`，设置 `MASTER_ADDR`（rank0可达地址）和 `MASTER_PORT`；JAX默认使用 `MASTER_PORT+1`，或显式 `JAX_COORDINATOR_ADDRESS=host:port`。入口按 deployment.training.num_nodes 设置进程数。两节点使用相同的代码、配置和共享数据路径。调度器配置仍需保证实际卡数与部署一致。

`run.rounds` 按最旧→最新填写，必须与索引的 collection_round 标签一致。最新轮占60%，历史40%递归分配；例如两轮40:60，三轮16:24:60。控制来源和结果采用相对权重，权重比不等于最终采样占比。

## 5. 评测

```bash
bash scripts/eval.sh --set run.name=final_eval \
  --set run.checkpoint=/mnt/esi/training/round3_train/checkpoints/6000
bash scripts/eval.sh --config configs/eval/baseline.json --set run.name=baseline_eval
```

默认评测使用 `configs/eval/final.json`，基准评测使用 `configs/eval/baseline.json`。设备、账号与权重位置在部署配置中设置。修改实验配置后，请使用新的实验名。

## 6. 可选 prompt 迭代

默认不自动运行。开启某轮采集后触发：`--set config.prompt_evolution.enabled=true`。独立分析已完成采集：

```bash
bash scripts/evolve.sh --set run.name=round1_prompt_review \
  --set run.suite=/mnt/esi/collection_suites/round1_collection
```

显式执行 evolve 表示请求分析，不受自动开关 false 限制。采用持续主分析会话与独立审核会话。默认生成审核通过的下一轮 bundle，不覆盖当前源文件，也不会自动启动下一轮。选择新 bundle 时在后续采集配方设置 `config.prompt_bundle` 和 `config.prompt_bundle_sha256`，同时设置新实验名和新权重。详见 [prompt 迭代说明](../hybrid_rollout/robodojo/prompt_evolution/README.md)。

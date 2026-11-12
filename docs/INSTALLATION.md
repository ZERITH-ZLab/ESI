# 安装与验证

本指南把 ESI 的源码、模拟器和 OpenPI 分开安装。命令中的路径由部署者选择；示例仅用于说明目录关系。所有实验输出、权重、资产和下载缓存建议放在大容量存储。不要在正在运行的实验所用外部源码或 Python 环境中执行升级。

## 基础环境与容器

推荐从 **Linux x86_64、Ubuntu 22.04、CUDA Toolkit 12.8** 开始安装。容器建议选择 `devel` 类型以提供编译工具，再按下文创建 Python 3.11 环境并安装依赖；仅有 CUDA 基础镜像不能直接运行 ESI。

Isaac Sim 5.1.0 的 pip 安装要求 Python 3.11 和 glibc ≥2.35。其官方系统列表也包括 Ubuntu 24.04，但这不等于任意 Ubuntu 24.04 镜像均可直接使用本安装配方。见 [Isaac Sim Python 安装要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。

模拟器节点需要支持 RTX 渲染的 GPU；无 RT Cores 的 A100/H100 不在 Isaac Sim 的支持范围内。模型推理和训练可部署在独立的计算 GPU 节点。驱动版本应同时满足所选 GPU、CUDA 和 Isaac Sim 的兼容要求，参照 [Isaac Sim 硬件与驱动要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)选择，不要只按 JAX 的最低驱动要求配置模拟器。

CUDA Toolkit、Python 包的 CUDA 运行库和宿主机驱动是不同组件。本安装配方的模拟器环境使用 PyTorch cu128，OpenPI 使用 JAX CUDA 12 包；它们可安装各自的运行库，不要求全部链接系统 Toolkit。宿主机仍必须提供兼容的 NVIDIA 驱动，安装容器内的 CUDA 包不能替代驱动。

使用 Docker 时，宿主机需安装 NVIDIA Container Toolkit，并在创建容器时分配 GPU，例如使用 `--gpus`。模拟器容器还需传入 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`；使用 GPU 视频编解码或窗口显示时分别增加 `video` 或 `display`。只开放计算能力不足以支持相机渲染，headless 模式也需要图形支持。详见 [NVIDIA 容器 GPU 与驱动能力配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)。

安装前检查 `ldd --version`、`nvcc --version` 和 `nvidia-smi`；容器内也应能看到分配的 GPU。`nvidia-smi` 显示的 CUDA 版本表示驱动支持能力，不等于已安装的 Toolkit 版本。依赖安装后再验证模拟器相机渲染和 π05 推理，两者分别通过后开始完整评测。

## 推荐参考布局

下面的布局与默认部署路径对应。先确保这些目录由部署用户读写；没有 `/opt` 或 `/mnt` 写权限时可以整体换到自己的目录，再修改 `configs/deployment.local.json`。路径不是用户名或硬件要求。

```text
控制/策略节点
/opt/
├── ESI/                                      # 本仓库
├── RoboDojo/                                 # 固定版本的外部源码
│   ├── Assets -> /mnt/robodojo/Assets
│   ├── XPolicyLab/
│   │   └── policy/Pi_05/openpi/
│   │       └── .venv/                         # 独立 π05 Python 环境
│   └── third_party/
│       ├── IsaacLab/
│       └── curobo/
└── envs/RoboDojo/                             # 模拟器/控制工具 Python
/mnt/
├── robodojo/Assets/                           # 下载的完整仿真资产
├── models/esi-final/                          # 最终版 checkpoint
├── esi/                                      # 实验共享输出
└── esi-private/                              # 私有认证和账号池

模拟器节点
/opt/RoboDojo/                                 # 同版本源码、补丁和依赖
/opt/envs/RoboDojo/                            # 模拟器环境
/mnt/robodojo/Assets/                          # 同版本完整资产
/mnt/esi/                                     # 与控制节点同路径、同一份共享存储
```

模拟器节点无需安装 π05 权重或 GPU 训练环境；控制节点需要 RoboDojo 源码用于任务配置与身份检查。单机部署可以合并两种角色，入口仍通过配置的 SSH 连接启动模拟器。依赖内部目录保留上游结构，不随意移动 OpenPI 的 `src` 或 `packages/openpi-client`。

以下安装完成后，只做最终 eval 的用户转到 [EVALUATION.md](EVALUATION.md)，不需要先下载演示数据、标注或训练。基准 eval/首轮采集才需要额外准备官方预训练权重。

## 1. 外部源码版本

在新目录中执行，先将 `ESI_SOURCE` 设为本仓库绝对路径，将 `EXTERNAL_ROOT` 设为外部源码目录：

```bash
export ESI_SOURCE=/opt/ESI
export EXTERNAL_ROOT=/opt
mkdir -p "$EXTERNAL_ROOT"
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "$EXTERNAL_ROOT/RoboDojo"
export ROBODOJO_SOURCE="$EXTERNAL_ROOT/RoboDojo"
git -C "$ROBODOJO_SOURCE" checkout --detach 08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d
git -C "$ROBODOJO_SOURCE" submodule update --init XPolicyLab third_party/IsaacLab third_party/curobo
git -C "$ROBODOJO_SOURCE/XPolicyLab" checkout --detach bb9a0b5f5136a74503b679af830bfd0a3a837d5c
git -C "$ROBODOJO_SOURCE/third_party/IsaacLab" checkout --detach afca7b09d60d8beb9c1cb28b43066499940b969b
git -C "$ROBODOJO_SOURCE/third_party/curobo" fetch origin 895c6517243f8cb091c73c018c8167192d39599a
git -C "$ROBODOJO_SOURCE/third_party/curobo" checkout --detach 895c6517243f8cb091c73c018c8167192d39599a
```

子模块 URL 来自该 RoboDojo commit 的 `.gitmodules`。使用 fork 而非同名上游仓库。后续不要执行 `git submodule update --remote`，也不要重新用父仓库 gitlink 覆盖上述 cuRobo 版本。

按 [补丁说明](../patches/robodojo/README.md) 先 `git apply --check`，再应用两份补丁。资产路径补丁适配 Assets 符号链接；诊断补丁保留原生规划失败信息。它们不修改任务判定、控制轨迹或碰撞阈值。OpenPI 不需要另外复制补丁。

## 2. 模拟器环境

按 [RoboDojo 官方安装说明](https://robodojo-benchmark.com/doc/usage/install-and-download/) 准备 Linux、NVIDIA 驱动、系统依赖、Isaac Sim 及其授权。参考环境使用 Python 3.11、Isaac Sim 5.1.0、PyTorch 2.7.0/cu128、SciPy 1.15.3 和 warp-lang 1.11.0。IsaacLab/cuRobo 必须安装上一步选定的源码。

**注意版本脚本的行为：**上述 RoboDojo 的 `scripts/install.sh -i` 和 `--from base_deps` 会执行 `setup_submodules` 并更新远程分支，破坏固定版本。部署者可按该脚本分别安装系统/基础包，选定源码后从 `--from isaacsim` 开始安装后续阶段。该脚本还会自动激活名为 `RoboDojo` 的 Conda 环境；自定义环境名时应按照其安装步骤手动执行，不能以为当前激活环境会一直生效。

例如在尚未安装 RoboDojo 的新容器中，先安装 `cmake`、`build-essential`、`ffmpeg` 和 Conda，再执行下面的步骤。它使用上游脚本要求的环境名，不适用于已经有同名运行环境的机器：

```bash
export CONDA_ENVS_PATH=/opt/envs
conda create -n RoboDojo python=3.11 -y
conda activate RoboDojo
cd "$ROBODOJO_SOURCE"
python -m pip install -r scripts/requirements.txt
python -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib \
  scipy==1.15.3 scikit-learn numpy==1.26.0
bash scripts/install.sh --from isaacsim
export SIM_PYTHON="$(command -v python)"
# 安装后复核源码版本，不能因安装脚本行为而悄悄切换。
git -C third_party/IsaacLab rev-parse HEAD
git -C third_party/curobo rev-parse HEAD
git -C XPolicyLab rev-parse HEAD
```

安装完上游环境后，在模拟器解释器中添加 ESI 所需的 CPU 碰撞库（不要升级整个环境）：

```bash
# SIM_PYTHON 指向新安装的模拟器 Python。
"$SIM_PYTHON" -m pip install 'Cython==3.0.12'
"$SIM_PYTHON" -m pip install --no-deps -r "$ESI_SOURCE/hybrid_rollout/robodojo/requirements-motion.txt"
"$SIM_PYTHON" -m pip check
```

控制节点还需安装 OpenPI client 的依赖，以便与 π05 服务通信。使用上一步的解释器，限制 NumPy 和 websockets 不被客户端升级；不要在这里安装整个 OpenPI：

```bash
"$SIM_PYTHON" -m pip install \
  "$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi/packages/openpi-client" \
  'numpy==1.26.0' 'websockets==12.0'
"$SIM_PYTHON" -m pip install -r "$ESI_SOURCE/requirements-tools.txt" \
  -r "$ESI_SOURCE/data_processing/requirements.txt" \
  'numpy==1.26.0' 'scipy==1.15.3' 'websockets==12.0'
"$SIM_PYTHON" -m pip check
```

`--no-deps` 是为了保留上游已安装的 NumPy/SciPy；Cython 是 python-fcl 声明依赖，不能遗漏。另按所用阶段安装 `requirements-tools.txt` 和 `data_processing/requirements.txt` 的工具包，先检查依赖解析结果，避免把 OpenPI/LeRobot 整套依赖装入模拟器环境。

上游参考环境仍有依赖声明冲突风险：IsaacLab 对 starlette 的要求可能不同于 RoboDojo 安装脚本，Isaac Sim 要求的 websockets 版本也不能随工具包升级。应记录 `pip check` 的结果并在新环境中验证；不要把参考环境包列表视为完整、无冲突的环境锁。

## 3. Assets 与配置生成

按 RoboDojo 官方下载脚本获取完整资产。权重另存，不能只靠本仓库的派生碰撞文件启动模拟器。以下仅在新的外部安装里建立链接；若 `Assets` 已存在，先确认其内容，不要覆盖：

```bash
# 使用已安装的 Hugging Face CLI，仅下载仿真 Assets，不下载训练数据。
# 可添加 --revision 指定相应上游资产版本；下方碰撞预检仍严格核对。
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --include 'Assets/**' --local-dir /mnt/robodojo
export ASSETS_DIR=/mnt/robodojo/Assets
ln -s "$ASSETS_DIR" "$ROBODOJO_SOURCE/Assets"
cd "$ROBODOJO_SOURCE"
"$SIM_PYTHON" utils/update_embodiment_config_path.py
```

此命令会从 `*_tmp.yml` 生成资产配置，所以应在自有可写资产目录执行；不要修改其他正在运行实验共享的配置。路径变化后重新生成，不能让 cuRobo YAML 仍指向原机器目录。

检查 X5 配置路径，再运行 ESI 的真实几何预检：

```bash
cd "$ESI_SOURCE"
"$SIM_PYTHON" - "$ASSETS_DIR/Robots/x5" <<'PY'
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1]).resolve()
k = yaml.safe_load((root/'curobo.yml').read_text())['robot_cfg']['kinematics']
assert Path(k['asset_root_path']).resolve() == root
assert Path(k['urdf_path']).resolve() == (root/'X5A.urdf').resolve()
print('X5 configuration paths match the selected assets')
PY
"$SIM_PYTHON" -m hybrid_rollout.robodojo.robodojo_server.cooked_collision \
  --asset-root "$ASSETS_DIR/Robots/x5"
```

预检会检查源文件身份、FK 与 cooked 几何坐标、FCL 后端的相交/分离对照。仅修改路径或 YAML 排版可以通过；实际几何或碰撞参数变化仍会拒绝。更换机器人不属于重新填写安装路径，需要重新适配碰撞模型。

## 4. OpenPI 环境

OpenPI 是该 XPolicyLab checkout 内的目录，不要另拉任意 OpenPI 主分支替换。按照固定版本中的 `policy/Pi_05/install.sh` 安装独立环境：

```bash
export OPENPI_SOURCE="$ROBODOJO_SOURCE/XPolicyLab/policy/Pi_05/openpi"
cd "$OPENPI_SOURCE"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --group lerobot
uv pip install -e .
uv pip install -e "$ROBODOJO_SOURCE/XPolicyLab"
uv pip check
.venv/bin/python -c 'import jax; from openpi.training import config; print(jax.devices()); print(config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0").name)'
```

参考训练环境：JAX/jaxlib 0.5.3、Flax 0.10.2、Optax 0.2.4、Orbax 0.11.13、NumPy 1.26.4、LeRobot 0.4.4。固定 checkout 的 `pyproject.toml`/`uv.lock` 是安装起点，不把已有环境的 `pip freeze` 当通用锁。该依赖组合中 rerun-sdk 可能声明 NumPy ≥2，而 OpenPI 要求 NumPy <2；这不应通过直接升级 NumPy 解决。先隔离验证需要的训练/导出功能，记录冲突；并用小规模运行验证所需功能。

推理 checkpoint（包括最终 ESI）目录必须含 `params/` 和 `assets/arx_x5_sim/norm_stats.json`。ESI 训练输出另有 `train_state/` 与进度元数据；不要将两类目录粒度混用。

## 5. 不依赖已有 Python 环境的 CPU 验证

从 ESI 根目录运行，使用全新 venv，不启用 `--system-site-packages`：

```bash
python3.11 -m venv /path/to/esi-validation-venv
export VALIDATION_PYTHON=/path/to/esi-validation-venv/bin/python
"$VALIDATION_PYTHON" -m pip install -r requirements-validation.txt
"$VALIDATION_PYTHON" -m pip check
"$VALIDATION_PYTHON" -m pytest -q \
  hybrid_rollout/robodojo/robodojo_server/test_collision_assets.py \
  hybrid_rollout/robodojo/robodojo_server/test_self_collision.py \
  release_tools/tests/test_entrypoints.py \
  hybrid_rollout/robodojo/test_eval_snapshot.py
"$VALIDATION_PYTHON" -m hybrid_rollout.robodojo.robodojo_server.cooked_collision \
  --asset-root "$ASSETS_DIR/Robots/x5"
```

`requirements-validation.txt` 只用于 CPU 资产/几何与入口配置验证，不是完整部署依赖或 GPU 训练锁。它不安装 Isaac Sim、cuRobo、OpenPI，也不会启动模拟器。

## 6. 配置入口与 GPU 验收

填写 `configs/deployment.local.json` 后，按 [CONFIGURATION.md](CONFIGURATION.md) 查看配置并逐阶段短测：采集少量执行步、导出与索引、两次训练更新后保存、另一个进程恢复两次更新、最终/基准各一个短 eval。验证实际选中的 Python、源文件和输出目录，而不仅是命令能解析。

本机与模拟器节点均需应用对应版本和补丁。SSH 用户由 `deployment.local.json` 的 `paths.sim_ssh_user` 设置，槽位可用 `ROLLOUT_SIM_SSH_USER` 覆盖；需相同路径的共享实验存储。8/16 卡是部署预设，只有实际多卡运行才能验证通信、分片和吞吐。CPU 检查不替代实际 GPU 运行检查。

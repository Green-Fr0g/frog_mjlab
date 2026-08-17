# frog-mjlab

基于 [mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（本地仓库 `mjlab`）的**外部扩展**训练框架，
参照 `unitree_rl_mjlab` / `AMP_mjlab` 的组织方式，从 mjlab 抽离出多个开箱即用的训练任务：

- **G1 速度跟踪（locomotion）**：G1 行走 / 跑步
- **H2 速度跟踪**：H2 行走 / 跑步
- **动作模仿（mimic / tracking）**：G1 模仿参考动作序列（BeyondMimic 风格）

- 基于 `mjlab.rl` 的 PPO 训练（`MjlabOnPolicyRunner` / `MotionTrackingOnPolicyRunner`）
- 训练与回放管线一致，训练/回放时自动导出 ONNX 策略
- 支持粗糙地形（Rough）与平地（Flat）两种配置

## 环境要求

- Linux
- Python 3.13（与 mjlab 一致）
- 可用的 MuJoCo 与 GPU 驱动（训练前置条件）
- 已安装 `mjlab`（本工程在 `mjlab` 仓库的 `.venv` 下运行）

> 说明：本工程已将 `update_assets` 这类旧版 mjlab 接口替换为当前 mjlab 的
> `mujoco.MjSpec.from_file` 资产加载方式，可直接在本地 `mjlab` 版本上运行。

## 安装

在已安装 mjlab 的虚拟环境中可编辑安装本工程（若 mjlab 已可用，建议加 `--no-deps` 避免重装）：

```bash
cd frog_mjlab
python -m pip install -e . --no-deps
```

或者不安装，直接通过 `PYTHONPATH` 运行（本项目开发时常用方式）：

```bash
export PYTHONPATH="$PWD/src"
```

## 列出可用任务

```bash
python scripts/list_envs.py --keyword G1
```

本工程注册的任务（G1 有 29 DoF 与 23 DoF 两个不同变体，任务 ID 已标注 DoF 数）：

**速度跟踪（locomotion）**
- `Unitree-G1-29-Rough` — G1 **29 DoF** 粗糙地形速度跟踪（使用 `g1.xml`）
- `Unitree-G1-29-Flat` — G1 **29 DoF** 平地速度跟踪（使用 `g1.xml`）
- `Unitree-H2-Rough` — H2 粗糙地形速度跟踪（使用 `h2.xml`）
- `Unitree-H2-Flat` — H2 平地速度跟踪（使用 `h2.xml`）

**动作模仿（mimic / tracking）**
- `Unitree-G1-Tracking` — G1 动作模仿（含状态估计）
- `Unitree-G1-Tracking-No-State-Estimation` — G1 动作模仿（不含状态估计）
- `Unitree-G1-23Dof-Tracking` — G1 23 DoF 动作模仿
- `Unitree-G1-23Dof-Tracking-No-State-Estimation` — G1 23 DoF 动作模仿（不含状态估计）

> 说明：G1 的 23 DoF 变体（`g1_23dof.xml`）未注册为速度跟踪任务，因为其模型缺少
> `left_foot`/`right_foot` 站点，无法复用 velocity 任务的 foot 观测/奖励；但它可用作动作模仿任务。

## 训练

**速度跟踪**

```bash
python scripts/train.py Unitree-G1-29-Flat --env.scene.num-envs=4096
python scripts/train.py Unitree-H2-Flat --env.scene.num-envs=4096
```

**动作模仿（mimic）**

先准备动作文件（已内置示例 `src/frog_mjlab/assets/motions/g1/dance1_subject2.csv`），
转成 npz（示例 npz 已预生成在同目录）：

```bash
python scripts/csv_to_npz.py \
  --input-file src/frog_mjlab/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 --output-fps 50 --robot g1
```

然后训练：

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/frog_mjlab/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096
```

训练日志默认保存到：

- `logs/rsl_rl/g1_locomotion/<time_stamp_run>/`
- `logs/rsl_rl/g1_tracking/<time_stamp_run>/`
- `logs/rsl_rl/h2_locomotion/<time_stamp_run>/`

## 回放 / 可视化

```bash
python scripts/play.py Unitree-G1-29-Rough \
  --checkpoint-file logs/rsl_rl/g1_locomotion/<run_dir>/model_<iter>.pt
```

回放默认启用 ONNX 导出，生成 `policy.onnx`。

## 仓库结构

```
model/
├── g1/                        # G1 机器人模型（MJCF XML + STL mesh）
│   ├── g1.xml                 # 机器人模型（29 DoF），训练代码使用
│   ├── g1_23dof.xml           # 23 DoF 变体
│   ├── scene_g1.xml           # 场景文件（机器人 + 地面/相机，供独立可视化）
│   ├── scene_g1_23dof.xml     # 23 DoF 场景变体
│   └── assets/                # STL mesh 网格
└── h2/                        # H2 机器人模型
    ├── h2.xml
    └── assets/                # STL mesh 网格
src/frog_mjlab/
├── __init__.py                 # SRC_PATH / MODEL_PATH
├── assets/
│   ├── g1/                     # G1 机器人常量与执行器配置（模型文件在 model/g1/）
│   ├── h2/                     # H2 机器人常量与执行器配置（模型文件在 model/h2/）
│   └── motions/g1/             # 动作模仿数据（csv + npz）
├── tasks/
│   ├── locomotion/             # 速度跟踪任务（G1 / H2）
│   │   ├── locomotion_env_cfg.py # 任务配置工厂
│   │   ├── mdp/                # rewards / observations / terminations / curriculum
│   │   ├── rl/runner.py        # LocomotionOnPolicyRunner（含 ONNX 导出）
│   │   └── config/g1_29/, config/h2/  # 各机器人环境/RL 配置 + 任务注册
│   └── tracking/               # 动作模仿（mimic）任务
│       ├── tracking_env_cfg.py # 任务配置工厂
│       ├── mdp/                # 动作模仿的 rewards / observations / terminations
│       ├── rl/runner.py        # MotionTrackingOnPolicyRunner（含 ONNX 导出）
│       └── config/g1/, config/g1_23dof/  # 注册
scripts/
├── train.py                    # 训练入口
├── play.py                     # 回放入口
├── list_envs.py                # 列出已注册任务
└── csv_to_npz.py               # 动作 CSV -> NPZ 转换工具
```

## 任务注册机制

`src/frog_mjlab/tasks/__init__.py` 通过 mjlab 的 `import_packages` 递归导入各任务模块，
`config/g1/__init__.py` 调用 `register_mjlab_task(...)` 将任务注册进 mjlab 的任务注册表，
从而复用 mjlab 的 `load_env_cfg` / `load_rl_cfg` / `load_runner_cls` / `list_tasks` 等工具。

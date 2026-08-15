# frog-mjlab

基于 [mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（本地仓库 `mjlab`）的**外部扩展**训练框架，
参照 `AMP_mjlab` 的组织方式，从 mjlab 抽离出一个开箱即用的 **G1 机器人速度跟踪（velocity）训练任务**。

- 单一策略学习 G1 行走 / 跑步（速度跟踪）
- 基于 `mjlab.rl` 的 PPO 训练（`MjlabOnPolicyRunner`）
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

本工程注册的任务：

- `Unitree-G1-Rough` — 粗糙地形速度跟踪
- `Unitree-G1-Flat` — 平地速度跟踪

## 训练

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

训练日志默认保存到：

- `logs/rsl_rl/g1_velocity/<time_stamp_run>/`

## 回放 / 可视化

```bash
python scripts/play.py Unitree-G1-Rough \
  --checkpoint-file logs/rsl_rl/g1_velocity/<run_dir>/model_<iter>.pt
```

回放默认启用 ONNX 导出，生成 `policy.onnx`。

## 仓库结构

```
model/
└── unitree_g1/xmls/            # 机器人模型资产（MJCF XML + STL mesh）
src/frog_mjlab/
├── __init__.py                 # SRC_PATH / MODEL_PATH
├── assets/
│   └── robots/unitree_g1/      # G1 机器人常量与执行器配置（模型文件在 model/）
├── tasks/
│   └── velocity/               # velocity 速度跟踪任务
│       ├── velocity_env_cfg.py # 任务配置工厂
│       ├── mdp/                # rewards / observations / terminations / curriculum
│       ├── rl/runner.py        # VelocityOnPolicyRunner（含 ONNX 导出）
│       └── config/g1/          # G1 环境与 RL 配置 + 任务注册
scripts/
├── train.py                    # 训练入口
├── play.py                     # 回放入口
└── list_envs.py                # 列出已注册任务
```

## 任务注册机制

`src/frog_mjlab/tasks/__init__.py` 通过 mjlab 的 `import_packages` 递归导入各任务模块，
`config/g1/__init__.py` 调用 `register_mjlab_task(...)` 将任务注册进 mjlab 的任务注册表，
从而复用 mjlab 的 `load_env_cfg` / `load_rl_cfg` / `load_runner_cls` / `list_tasks` 等工具。

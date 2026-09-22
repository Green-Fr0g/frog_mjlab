# frog-mjlab

基于 [mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（本地仓库 `mjlab`）的**外部扩展**训练框架，
参照 `frog_lab` 的目录布局与 `unitree_rl_mjlab` / `AMP_mjlab` 的任务实现，从 mjlab 抽离出多个开箱即用的训练任务：

- **G1 速度跟踪（locomotion）**：G1 行走 / 跑步
- **H2 速度跟踪**：H2 行走 / 跑步
- **动作模仿（mimic / tracking）**：G1 模仿参考动作序列（BeyondMimic 风格）
- **AMP 动作模仿**：G1 AMP（对抗式运动先验）行走 / 恢复

- 基于 `mjlab.rl` 的 PPO / AMP 训练（`LocomotionOnPolicyRunner` / `MotionMimicOnPolicyRunner` / `AMPOnPolicyRunner`）
- 算法包 `frog_rl`（源自 AMP_mjlab 的 vendored RSL-RL，含 `AMPPPO` / `AmpOnPolicyRunner`）
- 训练与回放管线一致，训练/回放时自动导出 ONNX + TorchScript 策略
- 支持粗糙地形（Rough）与平地（Flat）两种配置

## 环境要求

- Linux
- Python 3.13（与 mjlab 一致）
- 可用的 MuJoCo 与 GPU 驱动（训练前置条件）
- 已安装 `mjlab`（本工程在 `mjlab` 仓库的 `.venv` 下运行）

> 说明：本工程已将 `update_assets` 这类旧版 mjlab 接口替换为当前 mjlab 的
> `mujoco.MjSpec.from_file` 资产加载方式，可直接在本地 `mjlab` 版本上运行。

## 安装

布局对齐 `frog_lab`（`source/` 下两个可编辑安装的 Python 包）：

```bash
cd frog_mjlab
python -m pip install -e source/frog_mjlab --no-deps
python -m pip install -e source/frog_rl    --no-deps
```

或者不安装，直接通过 `PYTHONPATH` 运行（本项目开发时常用方式）：

```bash
export PYTHONPATH="$PWD/source/frog_mjlab:$PWD/source/frog_rl"
```

## 列出可用任务

```bash
python scripts/list_envs.py --keyword FrogMjlab
```

本工程注册的任务（G1 有 29 DoF 与 23 DoF 两个不同变体）：

**速度跟踪（locomotion）**
- `FrogMjlab-G1-Rough` — G1 **29 DoF** 粗糙地形速度跟踪（使用 `g1.xml`）
- `FrogMjlab-G1-Flat` — G1 **29 DoF** 平地速度跟踪（使用 `g1.xml`）
- `FrogMjlab-H2-Rough` — H2 粗糙地形速度跟踪（使用 `h2.xml`）
- `FrogMjlab-H2-Flat` — H2 平地速度跟踪（使用 `h2.xml`）

**AMP 动作模仿**
- `FrogMjlab-G1-AMP` — G1 AMP 平地

**动作模仿（mimic / tracking）**
- `FrogMjlab-G1-Mimic` — G1 动作模仿（含状态估计）
- `FrogMjlab-G1-Mimic-No-State-Estimation` — G1 动作模仿（不含状态估计）
- `FrogMjlab-G1-23Dof-Mimic` — G1 23 DoF 动作模仿
- `FrogMjlab-G1-23Dof-Mimic-No-State-Estimation` — G1 23 DoF 动作模仿（不含状态估计）

> 说明：G1 的 23 DoF 变体（`g1_23dof.xml`）未注册为速度跟踪任务，因为其模型缺少
> `left_foot`/`right_foot` 站点，无法复用 velocity 任务的 foot 观测/奖励；但它可用作动作模仿任务。

## 训练

**速度跟踪**

```bash
python scripts/train.py FrogMjlab-G1-Flat --env.scene.num-envs=4096
python scripts/train.py FrogMjlab-H2-Flat --env.scene.num-envs=4096
```

**动作模仿（mimic）**

转换脚本与 `frog_lab/scripts/mimic/` 结构对齐：**单文件转换器不读配置、纯命令行**，
再由 `config_csv_to_npz.py` 读取 yaml 作为配置驱动前端（逐条调用转换器）；
npz 会写入 `robot_name` / `joint_names` / `body_names` / `root_link_name` 元数据，
因此与 frog_lab 产出的 npz 可以互换使用。

原始 CSV 在 `motion_data/motion_tracking/<robot>/`，转换配置在 `motion_data/config/*.yaml`
（`motion_data.csv_paths` 列出待转换的一条或多条 CSV；输出名取 CSV 同名并自动去重）：

```bash
# G1 29 DoF
python scripts/mimic/config_csv_to_npz.py \
  --config motion_data/config/g1.yaml \
  --output_dir source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1/motions

# G1 23 DoF
python scripts/mimic/config_csv_to_npz.py \
  --config motion_data/config/g1_23dof.yaml \
  --output_dir source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1_23dof/motions
```

也可直接用单文件转换器（参数与 `frog_lab/scripts/mimic/csv_to_npz_frog.py` 一致，
额外参数如 `--device cpu` 可直接透传）：

```bash
python scripts/mimic/csv_to_npz.py \
  --input_file motion_data/motion_tracking/g1/G1_Take_102.bvh_60hz.csv \
  --input_fps 120 \
  --robot_name g1 \
  --root_link_name pelvis \
  --csv_joint_names left_hip_pitch_joint left_hip_roll_joint ... \
  --output_name source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1/motions/G1_Take_102.bvh_60hz.npz
```

回放转换后的动作（自动从 npz 的 `robot_name` 选择对应 mimic 任务）：

```bash
python scripts/mimic/replay_npz.py \
  --motion_file source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1/motions/dance1_subject2.npz
```

然后训练：

```bash
python scripts/frog_rl/train.py FrogMjlab-G1-Mimic-No-State-Estimation \
  --motion_file=source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1/motions/dance1_subject2.npz \
  --env.scene.num-envs=4096

# G1 23 DoF：npz 名取自 CSV 文件名，由上面的 config_csv_to_npz.py 生成
python scripts/frog_rl/train.py FrogMjlab-G1-23Dof-Mimic-No-State-Estimation \
  --motion_file=source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1_23dof/motions/dance1_subject2_from_g1_g1_23.npz \
  --env.scene.num-envs=4096
```

**AMP 动作模仿**（动作数据在 `source/frog_mjlab/frog_mjlab/tasks/amp/config/g1/motions/`，已内置 WalkandRun + Recovery npz）：

```bash
python scripts/frog_rl/train.py FrogMjlab-G1-AMP --env.scene.num-envs=4096
```

训练日志默认保存到：

- `logs/rsl_rl/g1_locomotion/<time_stamp_run>/`
- `logs/rsl_rl/g1_mimic/<time_stamp_run>/`
- `logs/rsl_rl/h2_locomotion/<time_stamp_run>/`
- `logs/rsl_rl/g1_amp_flat/<time_stamp_run>/`

## 回放 / 可视化

```bash
python scripts/frog_rl/play.py FrogMjlab-G1-Rough \
  --checkpoint-file logs/rsl_rl/g1_locomotion/<run_dir>/model_<iter>.pt
```

回放默认导出两种部署格式（`--export-onnx` / `--export-jit` 可分别关闭），生成：

- `logs/rsl_rl/<exp>/<run>/export/<task_id>_<ckpt>.onnx` — ONNX 部署图
- `logs/rsl_rl/<exp>/<run>/export/<task_id>_<ckpt>.pt` — TorchScript 部署图

训练时每个 checkpoint 也会同步产出 `policy.pt` + `policy.onnx`（`save_interval` 节奏，随 run 目录保存）。

### 部署参数导出（deploy.yaml）

训练启动时（rank 0），`scripts/frog_rl/train.py` 会自动调用
`frog_mjlab/utils/export_deploy_cfg.py`，在本次 run 的 `params/deploy.yaml`
中导出真机部署所需的全部环境侧参数，与 `policy.onnx` 配套构成完整部署包：

- `joint_ids_map`：实际使用的关节名列表（即策略动作序，无 SDK 重映射）
- `step_dt`：控制周期（物理步长 × decimation）
- `stiffness` / `damping` / `default_joint_pos` / `encoder_bias`：按关节序导出的执行器增益与默认状态
- `commands`：速度指令范围（mimic 等无速度指令的任务自动跳过）
- `actions` / `observations`：逐 term 的 scale、clip、offset、joint_ids、history_length、delay 参数

只导出 `actor` 观测组（策略输入）；critic/特权观测不导出。

## 仓库结构

```
model/                         # 机器人模型（MJCF XML + STL mesh）
├── g1/
└── h2/
motion_data/                   # 原始动作数据（raw CSV）+ 转换配置（config/*.yaml）
source/
├── frog_mjlab/                # 任务/环境包（可编辑安装）
│   ├── pyproject.toml, setup.py
│   └── frog_mjlab/
│       ├── __init__.py        # SRC_PATH / MODEL_PATH / MOTION_DATA_PATH
│       ├── assets/            # 机器人常量（g1/, h2/）+ 动作数据（motions/）
│       └── tasks/
│           ├── locomotion/    # 速度跟踪（G1 / H2）
│           ├── amp/           # AMP 动作模仿（含 ampmotion_loader, config/g1, mdp, rl）
│           └── mimic/      # mimic 动作模仿
└── frog_rl/                   # 算法包（源自 RSL-RL，含 AMPPPO / AmpOnPolicyRunner）
    ├── pyproject.toml, setup.py
    └── frog_rl/
        ├── algorithms/  (ppo, amp_ppo, amp_discriminator)
        ├── runners/     (on_policy_runner, amp_on_policy_runner)
        ├── storage/  modules/  networks/  env/  utils/
scripts/
├── frog_rl/  (train.py, play.py)   # 训练/回放入口
├── mimic/    (csv_to_npz.py, config_csv_to_npz.py, replay_npz.py)  # 动作 CSV -> NPZ 转换与回放
└── list_envs.py                     # 列出已注册任务
```

## 任务注册机制

`src/frog_mjlab/tasks/__init__.py` 通过 mjlab 的 `import_packages` 递归导入各任务模块，
`config/g1/__init__.py` 调用 `register_mjlab_task(...)` 将任务注册进 mjlab 的任务注册表，
从而复用 mjlab 的 `load_env_cfg` / `load_rl_cfg` / `load_runner_cls` / `list_tasks` 等工具。

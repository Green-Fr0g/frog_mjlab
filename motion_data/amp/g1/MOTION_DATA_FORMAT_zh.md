# AMP CSV 运动数据说明

本文档说明 `motion_data_csv/amp` 目录下 CSV 运动数据的动作类型、帧率、每列含义和复用注意事项。

## 数据位置

原始 CSV 数据位于：

```text
motion_data_csv/amp
```

这些 CSV 是机器人运动轨迹数据，每个文件是一段动作片段，每一行表示一帧。文件没有表头，全部为逗号分隔的数值。

## 动作类型

当前 CSV 文件覆盖的动作主要是 G1 机器人的 locomotion 动作，包括行走、慢跑、侧移、弧线运动和原地转向。

| 动作名 | 含义 |
| --- | --- |
| `walk_forward` | 向前走 |
| `walk_backward` | 向后走 |
| `walk_sideway_left` | 向左横移 |
| `walk_sideway_right` | 向右横移 |
| `walk_arc_cw` | 沿顺时针弧线走 |
| `jog_forward` | 向前慢跑 |
| `jog_backward` | 向后慢跑 |
| `jog_arc_cw` | 沿顺时针弧线慢跑 |
| `arc_walk_left` | 向左沿弧线行走 |
| `arc_jog_left` | 向左沿弧线慢跑 |
| `idle_turn_270` | 原地转 270 度 |
| `idle_turn_360` | 原地转 360 度 |
| `step_rotate_idle` | 原地迈步并转向或调整姿态 |

文件名中的 `loop` 表示动作片段适合循环播放或循环训练。

## 帧率

这批 CSV 在当前项目的转换流程中按原始帧率 `120 FPS` 读取。

转换脚本 `scripts/csv_to_npz.py` 会把原始 CSV 重采样为 `50 FPS` 的 NPZ 数据，用于训练和回放：

```bash
python scripts/csv_to_npz.py \
  --input-dir motion_data_csv/amp \
  --output-dir <output_dir> \
  --input-fps 120 \
  --output-fps 50
```

因此：

- 直接使用 CSV 时，建议按 `120 FPS` 解释。
- 使用转换后的 NPZ 时，项目默认按 `50 FPS` 解释。

## CSV 每列含义

每个 CSV 文件每行共有 `36` 列：

```text
root_pos(3) + root_quat_xyzw(4) + joint_pos(29)
```

具体如下：

| CSV 列号 | 字段 | 含义 | 单位 |
| --- | --- | --- | --- |
| 1 | `root_pos_x` | 根部或骨盆在世界坐标系下的 x 坐标 | m |
| 2 | `root_pos_y` | 根部或骨盆在世界坐标系下的 y 坐标 | m |
| 3 | `root_pos_z` | 根部或骨盆在世界坐标系下的 z 坐标 | m |
| 4 | `root_quat_x` | 根部姿态四元数 x 分量 | 无量纲 |
| 5 | `root_quat_y` | 根部姿态四元数 y 分量 | 无量纲 |
| 6 | `root_quat_z` | 根部姿态四元数 z 分量 | 无量纲 |
| 7 | `root_quat_w` | 根部姿态四元数 w 分量 | 无量纲 |
| 8-36 | `joint_pos` | 29 个关节角，顺序见下表 | rad |

注意：CSV 中四元数顺序是 `x, y, z, w`。项目转换脚本读入后会转换为 `w, x, y, z`，以适配后续 MuJoCo/训练代码中的格式。

## 关节列顺序

第 8-36 列对应 29 个关节角。列号和关节名对应关系如下：

| CSV 列号 | 关节名 | 中文说明 |
| --- | --- | --- |
| 8 | `left_hip_pitch_joint` | 左髋俯仰 |
| 9 | `left_hip_roll_joint` | 左髋横滚 |
| 10 | `left_hip_yaw_joint` | 左髋偏航 |
| 11 | `left_knee_joint` | 左膝 |
| 12 | `left_ankle_pitch_joint` | 左踝俯仰 |
| 13 | `left_ankle_roll_joint` | 左踝横滚 |
| 14 | `right_hip_pitch_joint` | 右髋俯仰 |
| 15 | `right_hip_roll_joint` | 右髋横滚 |
| 16 | `right_hip_yaw_joint` | 右髋偏航 |
| 17 | `right_knee_joint` | 右膝 |
| 18 | `right_ankle_pitch_joint` | 右踝俯仰 |
| 19 | `right_ankle_roll_joint` | 右踝横滚 |
| 20 | `waist_yaw_joint` | 腰部偏航 |
| 21 | `waist_roll_joint` | 腰部横滚 |
| 22 | `waist_pitch_joint` | 腰部俯仰 |
| 23 | `left_shoulder_pitch_joint` | 左肩俯仰 |
| 24 | `left_shoulder_roll_joint` | 左肩横滚 |
| 25 | `left_shoulder_yaw_joint` | 左肩偏航 |
| 26 | `left_elbow_joint` | 左肘 |
| 27 | `left_wrist_roll_joint` | 左腕横滚 |
| 28 | `left_wrist_pitch_joint` | 左腕俯仰 |
| 29 | `left_wrist_yaw_joint` | 左腕偏航 |
| 30 | `right_shoulder_pitch_joint` | 右肩俯仰 |
| 31 | `right_shoulder_roll_joint` | 右肩横滚 |
| 32 | `right_shoulder_yaw_joint` | 右肩偏航 |
| 33 | `right_elbow_joint` | 右肘 |
| 34 | `right_wrist_roll_joint` | 右腕横滚 |
| 35 | `right_wrist_pitch_joint` | 右腕俯仰 |
| 36 | `right_wrist_yaw_joint` | 右腕偏航 |

## 数据中没有的内容

原始 CSV 只包含：

- 根部位置
- 根部姿态
- 关节角

原始 CSV 不包含：

- 根部线速度
- 根部角速度
- 关节速度
- 接触状态
- 动作标签列
- 时间戳列

在本项目中，速度是在 CSV 转 NPZ 时根据帧率通过差分或插值后的梯度计算得到的。

## 复用建议

如果要把这些 CSV 用在别的项目中，建议按下面方式处理：

1. 按 `120 FPS` 读取原始 CSV。
2. 每一行作为一帧，不要把第一行当作表头。
3. 使用第 1-3 列作为根部世界坐标。
4. 使用第 4-7 列作为根部四元数，格式为 `x, y, z, w`。
5. 使用第 8-36 列作为关节角，单位为弧度。
6. 如果目标系统需要 `w, x, y, z` 四元数格式，需要重排四元数分量。
7. 如果目标机器人关节顺序不同，需要按关节名重新映射第 8-36 列。
8. 如果需要速度信息，需要根据帧率自行计算。

## 简单读取示例

```python
import numpy as np

csv_path = "motion_data_csv/amp/walk_forward_loop_002__A022.csv"
motion = np.loadtxt(csv_path, delimiter=",")

root_pos = motion[:, 0:3]       # x, y, z
root_quat_xyzw = motion[:, 3:7] # x, y, z, w
joint_pos = motion[:, 7:36]     # 29 joints, rad

# 如果后续系统需要 w, x, y, z：
root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

fps = 120.0
dt = 1.0 / fps
```


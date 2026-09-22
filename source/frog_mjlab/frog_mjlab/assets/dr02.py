"""DeepRobotics DR02-Pro constants."""

from pathlib import Path

import mujoco

from frog_mjlab import MODEL_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

DR02_XML: Path = MODEL_PATH / "DR02" / "xml" / "DR02_pro.xml"
assert DR02_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Meshes are loaded automatically from the xml's meshdir (model/DR02/meshes).
  return mujoco.MjSpec.from_file(str(DR02_XML))


##
# Actuator config.
##

# Large joints: hips (pitch/roll) and knee share the same drive.
DR02_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_y_joint",
    ".*_hip_x_joint",
    ".*_knee_joint",
  ),
  stiffness=250.0,
  damping=6.0,
  effort_limit=330.0,
)
DR02_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_z_joint",),
  stiffness=180.0,
  damping=4.0,
  effort_limit=105.0,
)
DR02_ACTUATOR_ANKLE_Y = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_y_joint",),
  stiffness=100.0,
  damping=2.5,
  effort_limit=105.0,
)
DR02_ACTUATOR_ANKLE_X = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_x_joint",),
  stiffness=40.0,
  damping=1.0,
  effort_limit=35.0,
)
# The waist drive is shared by all three waist joints; the upstream config only
# parameterizes waist_z, so waist_x/waist_y reuse the same gains.
DR02_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_z_joint",
    "waist_x_joint",
    "waist_y_joint",
  ),
  stiffness=150.0,
  damping=3.0,
  effort_limit=105.0,
)
# Shoulders, elbow and wrist share the arm drive.
DR02_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_y_joint",
    ".*_shoulder_x_joint",
    ".*_shoulder_z_joint",
    ".*_elbow_joint",
    ".*_wrist_z_joint",
    ".*_wrist_y_joint",
    ".*_wrist_x_joint",
  ),
  stiffness=100.0,
  damping=2.5,
  effort_limit=105.0,
)
# The neck drive is not parameterized upstream. The MJCF declares it with the same
# ctrlrange tier as the ankle_x drive (+-50 N*m), so the smallest drive of the robot
# is reused here.
DR02_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "neck_z_joint",
    "neck_y_joint",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=35.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    ".*_hip_y_joint": -0.1,
    ".*_hip_x_joint": 0.0,
    ".*_hip_z_joint": 0.0,
    ".*_knee_joint": 0.2,
    ".*_ankle_y_joint": -0.1,
    ".*_ankle_x_joint": 0.0,
    "waist_z_joint": 0.0,
    ".*_shoulder_y_joint": 0.0,
    "left_shoulder_x_joint": 0.15,
    "right_shoulder_x_joint": -0.15,
    ".*_shoulder_z_joint": 0.0,
    ".*_elbow_joint": 1.35,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

DR02_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DR02_ACTUATOR_HIP_KNEE,
    DR02_ACTUATOR_HIP_YAW,
    DR02_ACTUATOR_ANKLE_Y,
    DR02_ACTUATOR_ANKLE_X,
    DR02_ACTUATOR_WAIST,
    DR02_ACTUATOR_ARM,
    DR02_ACTUATOR_NECK,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dr02_robot_cfg() -> EntityCfg:
  """Get a fresh DR02 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=DR02_ARTICULATION,
  )


DR02_ACTION_SCALE: dict[str, float] = {}
for a in DR02_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    DR02_ACTION_SCALE[n] = 0.25 * e / s

##
# Joint and body names.
##

DR02_ROOT_LINK_NAME = "base_link"
DR02_ALL_JOINT_NAMES = (
  "waist_z_joint",
  "waist_x_joint",
  "waist_y_joint",
  "left_shoulder_y_joint",
  "left_shoulder_x_joint",
  "left_shoulder_z_joint",
  "left_elbow_joint",
  "left_wrist_z_joint",
  "left_wrist_y_joint",
  "left_wrist_x_joint",
  "right_shoulder_y_joint",
  "right_shoulder_x_joint",
  "right_shoulder_z_joint",
  "right_elbow_joint",
  "right_wrist_z_joint",
  "right_wrist_y_joint",
  "right_wrist_x_joint",
  "neck_z_joint",
  "neck_y_joint",
  "left_hip_y_joint",
  "left_hip_x_joint",
  "left_hip_z_joint",
  "left_knee_joint",
  "left_ankle_y_joint",
  "left_ankle_x_joint",
  "right_hip_y_joint",
  "right_hip_x_joint",
  "right_hip_z_joint",
  "right_knee_joint",
  "right_ankle_y_joint",
  "right_ankle_x_joint",
)
# The neck joints are actuated but excluded from the controlled joint set.
DR02_CONTROL_JOINT_NAMES = tuple(
  name
  for name in DR02_ALL_JOINT_NAMES
  if name not in {"neck_z_joint", "neck_y_joint"}
)
DR02_AMP_JOINT_NAMES = DR02_CONTROL_JOINT_NAMES
DR02_BODY_NAMES = (
  "base_link",
  "waist_z_link",
  "waist_x_link",
  "body",
  "left_shoulder_y_link",
  "left_shoulder_x_link",
  "left_shoulder_z_link",
  "left_elbow_link",
  "left_wrist_z_link",
  "left_wrist_y_link",
  "left_wrist_x_link",
  "right_shoulder_y_link",
  "right_shoulder_x_link",
  "right_shoulder_z_link",
  "right_elbow_link",
  "right_wrist_z_link",
  "right_wrist_y_link",
  "right_wrist_x_link",
  "neck_link",
  "head_link",
  "left_hip_y_link",
  "left_hip_x_link",
  "left_hip_z_link",
  "left_knee_link",
  "left_ankle_y_link",
  "left_ankle_x_link",
  "right_hip_y_link",
  "right_hip_x_link",
  "right_hip_z_link",
  "right_knee_link",
  "right_ankle_y_link",
  "right_ankle_x_link",
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_dr02_robot_cfg())

  viewer.launch(robot.spec.compile())

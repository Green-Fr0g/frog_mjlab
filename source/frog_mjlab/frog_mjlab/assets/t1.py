"""Booster T1 constants."""

from pathlib import Path

import mujoco

from frog_mjlab import MODEL_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

T1_XML: Path = MODEL_PATH / "t1" / "xml" / "T1_serial.xml"
assert T1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Meshes are loaded automatically from the xml's meshdir (model/t1/meshes).
  return mujoco.MjSpec.from_file(str(T1_XML))


##
# Actuator config.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

ARMATURE_ARM = 0.0282528
ARMATURE_WAIST = 0.0478125
ARMATURE_HIP_PITCH = 0.0523908
ARMATURE_KNEE = 0.095625
ARMATURE_ANKLE = 0.0339552
ARMATURE_HEAD = 0.0018

STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ**2
STIFFNESS_WAIST = ARMATURE_WAIST * NATURAL_FREQ**2
STIFFNESS_HIP_PITCH = ARMATURE_HIP_PITCH * NATURAL_FREQ**2
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
DAMPING_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_WAIST * NATURAL_FREQ
DAMPING_HIP_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_HIP_PITCH * NATURAL_FREQ
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ

T1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=38.3,
  armature=ARMATURE_ARM,
)
T1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("Waist",),
  stiffness=STIFFNESS_WAIST,
  damping=DAMPING_WAIST,
  effort_limit=68.0,
  armature=ARMATURE_WAIST,
)
T1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=STIFFNESS_HIP_PITCH,
  damping=DAMPING_HIP_PITCH,
  effort_limit=96.0,
  armature=ARMATURE_HIP_PITCH,
)
T1_ACTUATOR_HIP_ROLL_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Roll", ".*_Hip_Yaw"),
  stiffness=STIFFNESS_WAIST,
  damping=DAMPING_WAIST,
  effort_limit=68.0,
  armature=ARMATURE_WAIST,
)
T1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=130.0,
  armature=ARMATURE_KNEE,
)
# The parallel ankle linkage doubles the reflected armature.
T1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
  stiffness=STIFFNESS_ANKLE,
  damping=DAMPING_ANKLE,
  effort_limit=76.0,
  armature=2.0 * ARMATURE_ANKLE,
)
T1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Head.*",),
  stiffness=STIFFNESS_HEAD,
  damping=DAMPING_HEAD,
  effort_limit=7.0,
  armature=ARMATURE_HEAD,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.70),
  joint_pos={
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
    "Left_Elbow_Yaw": -0.5,
    "Right_Elbow_Yaw": 0.5,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_WAIST,
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_HIP_ROLL_YAW,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_ANKLE,
    T1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg() -> EntityCfg:
  """Get a fresh T1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=T1_ARTICULATION,
  )


T1_ACTION_SCALE: dict[str, float] = {}
for a in T1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    T1_ACTION_SCALE[n] = 0.25 * e / s

##
# Joint and body names.
##

T1_ROOT_LINK_NAME = "Trunk"
T1_ALL_JOINT_NAMES = (
  "AAHead_yaw",
  "Head_pitch",
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Waist",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
)
# The head joints are actuated but excluded from the controlled joint set.
T1_CONTROL_JOINT_NAMES = tuple(T1_ALL_JOINT_NAMES[2:])
T1_AMP_JOINT_NAMES = T1_CONTROL_JOINT_NAMES
T1_BODY_NAMES = (
  "Trunk",
  "H1",
  "H2",
  "AL1",
  "AL2",
  "AL3",
  "left_hand_link",
  "AR1",
  "AR2",
  "AR3",
  "right_hand_link",
  "Waist",
  "Hip_Pitch_Left",
  "Hip_Roll_Left",
  "Hip_Yaw_Left",
  "Shank_Left",
  "Ankle_Cross_Left",
  "left_foot_link",
  "Hip_Pitch_Right",
  "Hip_Roll_Right",
  "Hip_Yaw_Right",
  "Shank_Right",
  "Ankle_Cross_Right",
  "right_foot_link",
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_t1_robot_cfg())

  viewer.launch(robot.spec.compile())

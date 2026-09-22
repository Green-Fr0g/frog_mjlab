"""EngineAI PM01 constants."""

from pathlib import Path

import mujoco

from frog_mjlab import MODEL_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

PM01_XML: Path = MODEL_PATH / "pm01" / "xml" / "serial_pm01_edu.xml"
assert PM01_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Meshes are loaded automatically from the xml's meshdir (model/pm01/meshes).
  return mujoco.MjSpec.from_file(str(PM01_XML))


##
# Actuator config.
##

# Motor parameters from the official PM01 asset config.
ARMATURE_Q90 = 0.0453
EFFORT_LIMIT_Q90 = 164.0
ARMATURE_Q25 = 0.0067
EFFORT_LIMIT_Q25 = 52.0

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ

PM01_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_PITCH.*",),
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
)
PM01_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_ROLL.*",),
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
)
PM01_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_YAW.*",),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)
PM01_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_KNEE_PITCH.*",),
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
)
PM01_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*ANKLE.*",),
  stiffness=STIFFNESS_Q25,
  damping=0.5,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)
PM01_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("J12_WAIST_YAW",),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)
PM01_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*SHOULDER.*", ".*ELBOW.*"),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)
PM01_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("J23_HEAD_YAW",),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9),
  joint_pos={
    ".*_HIP_PITCH.*": -0.06,
    ".*_KNEE_PITCH.*": 0.12,
    ".*_ANKLE_PITCH.*": -0.06,
    ".*_ELBOW_PITCH.*": -0.25,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J19_SHOULDER_ROLL_R": -0.15,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

PM01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PM01_ACTUATOR_HIP_PITCH,
    PM01_ACTUATOR_HIP_ROLL,
    PM01_ACTUATOR_HIP_YAW,
    PM01_ACTUATOR_KNEE,
    PM01_ACTUATOR_ANKLE,
    PM01_ACTUATOR_WAIST,
    PM01_ACTUATOR_ARM,
    PM01_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_pm01_robot_cfg() -> EntityCfg:
  """Get a fresh PM01 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=PM01_ARTICULATION,
  )


PM01_ACTION_SCALE: dict[str, float] = {}
for a in PM01_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    PM01_ACTION_SCALE[n] = 0.25 * e / s

##
# Joint and body names.
##

PM01_ROOT_LINK_NAME = "LINK_BASE"
PM01_ALL_JOINT_NAMES = (
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_YAW",
)
# The head joint is actuated but excluded from the controlled joint set.
PM01_CONTROL_JOINT_NAMES = tuple(PM01_ALL_JOINT_NAMES[:-1])
PM01_AMP_JOINT_NAMES = PM01_CONTROL_JOINT_NAMES
PM01_BODY_NAMES = (
  "LINK_BASE",
  "LINK_HIP_PITCH_L",
  "LINK_HIP_ROLL_L",
  "LINK_HIP_YAW_L",
  "LINK_KNEE_PITCH_L",
  "LINK_ANKLE_PITCH_L",
  "LINK_ANKLE_ROLL_L",
  "LINK_HIP_PITCH_R",
  "LINK_HIP_ROLL_R",
  "LINK_HIP_YAW_R",
  "LINK_KNEE_PITCH_R",
  "LINK_ANKLE_PITCH_R",
  "LINK_ANKLE_ROLL_R",
  "LINK_TORSO_YAW",
  "LINK_SHOULDER_PITCH_L",
  "LINK_SHOULDER_ROLL_L",
  "LINK_SHOULDER_YAW_L",
  "LINK_ELBOW_PITCH_L",
  "LINK_ELBOW_YAW_L",
  "LINK_SHOULDER_PITCH_R",
  "LINK_SHOULDER_ROLL_R",
  "LINK_SHOULDER_YAW_R",
  "LINK_ELBOW_PITCH_R",
  "LINK_ELBOW_YAW_R",
  "LINK_HEAD_YAW",
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pm01_robot_cfg())

  viewer.launch(robot.spec.compile())

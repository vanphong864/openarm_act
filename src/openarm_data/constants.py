"""The OpenArm dataset contract, as openarm_mc's recorder writes it.

Sixteen joints in one fixed order: arm joints in radians, grippers as
finger stroke in metres. State and absolute action targets share this order.
"""

JOINT_NAMES = (
    "left_j1", "left_j2", "left_j3", "left_j4",
    "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_j7", "right_gripper",
)  # fmt: skip
NUM_JOINTS = len(JOINT_NAMES)
GRIPPERS = (JOINT_NAMES.index("left_gripper"), JOINT_NAMES.index("right_gripper"))
ARM_JOINTS = tuple(i for i in range(NUM_JOINTS) if i not in GRIPPERS)

# With data.delta_actions, arm joints become deltas from the current state and
# the grippers stay absolute (openpi's make_bool_mask(7, -1, 7, -1)).
DELTA_MASK = tuple(i not in GRIPPERS for i in range(NUM_JOINTS))

GRIPPER_MAX_STROKE_M = 0.043

# OpenArm V1 position limits (rad; gripper stroke m), in JOINT_NAMES order.
# From openarm_mc's src/openarm_mujoco/v1/openarm_bimanual.xml; the two arms
# mirror each other in j1 and j2. Gripper: openarm_can's full stroke.
V1_JOINT_LIMITS = (
    (-3.490659, 1.396263), (-3.316125, 0.174533), (-1.570796, 1.570796), (0.0, 2.443461),
    (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, GRIPPER_MAX_STROKE_M),
    (-1.396263, 3.490659), (-0.174533, 3.316125), (-1.570796, 1.570796), (0.0, 2.443461),
    (-1.570796, 1.570796), (-0.785398, 0.785398), (-1.570796, 1.570796), (0.0, GRIPPER_MAX_STROKE_M),
)  # fmt: skip

RECORDER_FPS = 50
# The recorder's frame size now. Earlier episodes are 640x480; a dataset
# keeps whatever size its episodes have, and the server tells the client.
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 424
# Legacy RGB-D preprocessing range: metres
# clipped to this range and scaled to 0..1; 0 (no reading) lands on the near end.
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 1.2

# Legacy single-camera observation names.
CAMERA_RGB = "chest"
CAMERA_DEPTH = "chest_depth"

# Where each part of an episode lives in the recorder's HDF5.
H5_RGB = "observations/images/chest_rgb"
H5_DEPTH = "observations/images/chest_depth"
H5_QPOS = "observations/qpos"
H5_ACTION = "action"
H5_COMMANDS = "joint_commands"
H5_TIMESTAMP = "timestamp_ns"


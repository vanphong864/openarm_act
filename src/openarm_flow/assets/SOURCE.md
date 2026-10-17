# OpenArm V1 kinematic constants

`openarm_v1_kinematics.json` contains fixed transforms, hinge axes/pivots, joint
references and limits exported from the OpenArm V1 model vendored at
`simulation/third_party/openarm_mujoco/v1/openarm_bimanual.xml`, as loaded by
the existing OpenArm simulator scene. It contains no meshes or dynamics.

Source model copyright/license: Apache-2.0, reproduced in `OPENARM_LICENSE`.
The JSON records the source XML SHA256. The exported transforms are used by
the new PyTorch FK implementation; tests compare both TCP poses with MuJoCo.
This is robot-model geometry, not independently calibrated real TCP ground truth.

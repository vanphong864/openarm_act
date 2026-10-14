# Source

Copied from `/home/space/VR/FlowPolicy`, upstream https://github.com/zql-kk/FlowPolicy,
commit `86d10c14972b4eab083abec9026749c340f8e5ed` (MIT; LICENSE alongside).

Only PointNet, conditional U-Net, consistency loss, normalizer, EMA and their
small dependencies are included. Imports use the `openarm_flow._vendor` namespace.
Removed process-wide warning suppression, path injection and unused torchvision.
The inference loop directly implements the upstream sigma_var=0 Euler update,
avoiding the unrelated SDE/scipy model stack. The consistency training objective
and model layers are retained. Corrected the `cond` keyword to `local_cond`.
OpenArm data conversion, training CLI and deployment live outside this directory.

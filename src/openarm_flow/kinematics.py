"""Small differentiable OpenArm V1 FK; no MuJoCo needed for training/serving.

The fixed geometry is exported from the vendored robot model. FK losses are
kinematic proxies, not independent measurements of the real end effector.
"""

import json
from pathlib import Path

import torch
from torch import nn


class OpenArmKinematics(nn.Module):
    def __init__(self):
        super().__init__()
        spec = json.loads((Path(__file__).parent / "assets/openarm_v1_kinematics.json").read_text())
        self.provenance = spec["provenance"]
        self.chains = []
        lower, upper = [0.] * 16, [0.] * 16
        for side in ("left", "right"):
            chain = []
            for i, node in enumerate(spec["arms"][side]):
                prefix = f"{side}_{i}"
                self.register_buffer(prefix + "_fixed", torch.tensor(node["fixed_transform"], dtype=torch.float32))
                joint = node["joint"]
                if joint:
                    axis = torch.tensor(joint["axis"], dtype=torch.float32)
                    x, y, z = axis.tolist()
                    skew = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float32)
                    self.register_buffer(prefix + "_skew", skew)
                    self.register_buffer(prefix + "_pivot", torch.tensor(joint["pivot"], dtype=torch.float32))
                    lower[joint["action_index"]], upper[joint["action_index"]] = joint["range"]
                chain.append((prefix, joint["action_index"] if joint else None, joint["reference"] if joint else 0.))
            self.chains.append(chain)
        for g in (7, 15):
            lower[g], upper[g] = 0., .043
        self.register_buffer("lower", torch.tensor(lower))
        self.register_buffer("upper", torch.tensor(upper))

    def clamp(self, qpos):
        return torch.maximum(self.lower, torch.minimum(self.upper, qpos))

    def forward(self, qpos, *, clamp=False):
        if qpos.shape[-1] != 16:
            raise ValueError("Expected 16 joint positions")
        if clamp:
            qpos = self.clamp(qpos)
        identity = torch.eye(3, device=qpos.device, dtype=qpos.dtype)
        positions, rotations = [], []
        for chain in self.chains:
            r = identity.expand(*qpos.shape[:-1], 3, 3)
            p = torch.zeros(*qpos.shape[:-1], 3, device=qpos.device, dtype=qpos.dtype)
            for prefix, index, reference in chain:
                fixed = getattr(self, prefix + "_fixed").to(qpos.dtype)
                p = p + (r @ fixed[:3, 3, None]).squeeze(-1)
                r = r @ fixed[:3, :3]
                if index is not None:
                    theta = qpos[..., index] - reference
                    skew = getattr(self, prefix + "_skew").to(qpos.dtype)
                    pivot = getattr(self, prefix + "_pivot").to(qpos.dtype)
                    rotation = identity + theta.sin()[..., None, None] * skew + (1 - theta.cos())[..., None, None] * (skew @ skew)
                    offset = pivot - (rotation @ pivot[:, None]).squeeze(-1)
                    p = p + (r @ offset[..., None]).squeeze(-1)
                    r = r @ rotation
            positions.append(p)
            rotations.append(r)
        return torch.stack(positions, dim=-2), torch.stack(rotations, dim=-3)

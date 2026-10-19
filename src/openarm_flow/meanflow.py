"""Conditional MeanFlow for OpenArm: two depth views -> one-step action chunks.

MeanFlow identity: u_target = v - (t-r) * (v.d_z u + d_t u), with
stop-gradient on the entire target. New implementation using existing licensed
FlowPolicy residual blocks. See docs/MEANFLOW.md for sources and deviations.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from openarm_flow._vendor.model.flow.conditional_unet1d import ConditionalResidualBlock1D
from openarm_flow._vendor.model.flow.conv1d_components import Downsample1d, Upsample1d
from openarm_flow._vendor.model.flow.positional_embedding import SinusoidalPosEmb


class SpatialSoftmaxHead(nn.Module):
    """Learned keypoints: per-channel soft-argmax over the feature map, in [-1,1] image coordinates.

    Keeps sub-cell position information that a coarse average pool throws away; the global average
    of the same map is concatenated so appearance/context is not lost.
    """
    def __init__(self, channels, keypoints, feature_dim):
        super().__init__()
        self.keypoints = nn.Conv2d(channels, keypoints, 1)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.out = nn.Sequential(nn.Linear(2 * keypoints + channels, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())

    def forward(self, x):
        n, _, h, w = x.shape
        logits = self.keypoints(x).flatten(2) * self.log_temperature.exp()
        weights = logits.softmax(-1).view(n, -1, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        px = (weights.sum(2) * xs).sum(-1)
        py = (weights.sum(3) * ys).sum(-1)
        return self.out(torch.cat([px, py, x.mean((2, 3))], dim=-1))


class ResNetKeypointTower(nn.Module):
    """ImageNet ResNet18 stem..layer2 (stride 8, 128 channels) + keypoint head, for normalized RGB input."""
    def __init__(self, keypoints, feature_dim, pretrained=True, freeze_stem=False):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
        # Frozen stem + layer1 (weights and BatchNorm statistics): generic low-level ImageNet features
        # are kept, so fewer parameters can fit appearance details specific to the training images.
        self.frozen = nn.ModuleList([net.conv1, net.bn1, net.layer1]) if freeze_stem else nn.ModuleList()
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.head = SpatialSoftmaxHead(128, keypoints, feature_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def forward(self, image):
        # Policy RGB is (x - .5) / .25 with x in [0, 1]; dropped views are 0 (= mid gray).
        return self.head(self.trunk(((image * .25 + .5) - self.mean) / self.std))


class DualDepthEncoder(nn.Module):
    """Shared small CNN; ordered per-view features retain camera identity and layout."""
    def __init__(self, feature_dim=96, state_dim=64, use_rgb=False, spatial_softmax=False, keypoints=32,
                 rgb_backbone="cnn", pretrained_backbone=True, freeze_backbone_stem=False):
        super().__init__()
        if rgb_backbone not in ("cnn", "resnet18"):
            raise ValueError("rgb_backbone must be 'cnn' or 'resnet18'")

        def tower(channels):
            if spatial_softmax:  # stride 4 total: 30x40 map for 120x160 input, then keypoints
                layers = []
                for width, stride in ((16, 2), (32, 2), (64, 1)):
                    layers += [nn.Conv2d(channels, width, 3, stride=stride, padding=1),
                               nn.GroupNorm(8, width), nn.SiLU()]
                    channels = width
                return nn.Sequential(*layers, SpatialSoftmaxHead(64, keypoints, feature_dim))
            layers = []
            for width in (16, 32, 64):
                layers += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                           nn.GroupNorm(8, width), nn.SiLU()]
                channels = width
            return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
                                 nn.Linear(64 * 3 * 4, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())

        self.cnn = tower(4)  # metric-depth encoding, mask, image u and v (not XYZ)
        self.rgb_resnet = use_rgb and rgb_backbone == "resnet18"
        if self.rgb_resnet:
            self.rgb_cnn = ResNetKeypointTower(keypoints, feature_dim, pretrained_backbone, freeze_backbone_stem)
        else:
            self.rgb_cnn = tower(5) if use_rgb else None  # normalized RGB + image u and v
        self.state = nn.Sequential(nn.Linear(16, state_dim), nn.SiLU(), nn.Linear(state_dim, state_dim))
        self.output_dim = 2 * feature_dim * (2 if use_rgb else 1) + state_dim

    @staticmethod
    def _coords(like, n):
        h, w = like.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=like.device, dtype=like.dtype),
                                torch.linspace(-1, 1, w, device=like.device, dtype=like.dtype), indexing="ij")
        return torch.stack([xx, yy])[None].expand(n, -1, -1, -1)

    def forward(self, depth, state, rgb=None):
        b, t, v, c, h, w = depth.shape
        if v != 2 or c != 2 or state.shape != (b, t, 16):
            raise ValueError("Expected depth [B,T,2,2,H,W] and state [B,T,16]")
        if (rgb is None) != (self.rgb_cnn is None):
            raise ValueError("RGB input must be given exactly when the encoder uses RGB")
        n = b * t * v
        features = [self.cnn(torch.cat([depth.reshape(n, c, h, w), self._coords(depth, n)], dim=1)).reshape(b, t, -1)]
        if rgb is not None:
            if rgb.shape[:3] != (b, t, 2) or rgb.shape[3] != 3:
                raise ValueError("Expected rgb [B,T,2,3,H,W]")
            image = rgb.reshape(n, 3, *rgb.shape[-2:])
            if not self.rgb_resnet:
                image = torch.cat([image, self._coords(image, n)], dim=1)
            features.append(self.rgb_cnn(image).reshape(b, t, -1))
        return torch.cat([*features, self.state(state)], dim=-1).flatten(1)


class MeanVelocityUNet(nn.Module):
    def __init__(self, cond_dim, widths=(64, 128, 256), time_dim=64, input_dim=16):
        super().__init__()
        self.time_t = nn.Sequential(SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.time_r = nn.Sequential(SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        cond_dim += 2 * time_dim

        def block(a, b):
            return ConditionalResidualBlock1D(a, b, cond_dim, kernel_size=3, n_groups=8, condition_type="film")

        self.down = nn.ModuleList()
        for i, width in enumerate(widths):
            self.down.append(nn.ModuleList([block(input_dim if i == 0 else widths[i - 1], width),
                                            block(width, width),
                                            Downsample1d(width) if i < len(widths) - 1 else nn.Identity()]))
        self.mid = nn.ModuleList([block(widths[-1], widths[-1]), block(widths[-1], widths[-1])])
        self.up = nn.ModuleList([nn.ModuleList([block(2 * widths[i], widths[i - 1]),
                                                block(widths[i - 1], widths[i - 1]), Upsample1d(widths[i - 1])])
                                 for i in range(len(widths) - 1, 0, -1)])
        self.output = nn.Conv1d(widths[0], input_dim, 1)

    def forward(self, z, t, r, cond, return_features=False):
        condition = torch.cat([self.time_t(t), self.time_r(r), cond], dim=-1)
        x, skips = z.transpose(1, 2), []
        for first, second, down in self.down:
            x = second(first(x, condition), condition)
            skips.append(x)
            x = down(x)
        features = tuple(x.flatten(1) for x in skips)
        for block in self.mid:
            x = block(x, condition)
        for first, second, up in self.up:
            x = up(second(first(torch.cat([x, skips.pop()], dim=1), condition), condition))
        velocity = self.output(x).transpose(1, 2)
        return (velocity, features) if return_features else velocity


def meanflow_target(model, z, t, r, velocity, condition):
    """Exact JVP along (v,1,0); condition is fixed along the flow trajectory.

    The target has no parameter graph. A separate ordinary forward below keeps
    gradients to BOTH the velocity model and the observation encoder, without
    backpropagating through second derivatives of the target.
    """
    with torch.enable_grad():
        _, derivative = torch.autograd.functional.jvp(
            lambda zz, tt, rr: model(zz, tt, rr, condition.detach()),
            (z.detach(), t.detach(), r.detach()),
            (velocity.detach(), torch.ones_like(t), torch.zeros_like(r)), create_graph=False)
    return (velocity - (t - r)[:, None, None] * derivative).detach()


def dispersive_loss(features):
    """Bounded feature repulsion, excluding self-pairs; finite for B=1/collapse."""
    if features[0].shape[0] < 2:
        return features[0].sum() * 0
    terms = []
    for feature in features:
        f = F.normalize(feature, dim=-1, eps=1e-6)
        distance = (2 - 2 * (f @ f.T)).clamp_min(0)
        pairs = -distance[~torch.eye(len(f), device=f.device, dtype=torch.bool)]
        terms.append(torch.logsumexp(pairs, dim=0) - math.log(pairs.numel()))
    return torch.stack(terms).mean()


class MeanFlowPolicy(nn.Module):
    def __init__(self, horizon=16, n_obs_steps=2, n_action_steps=8, down_dims=(64, 128, 256),
                 time_dim=64, feature_dim=96, state_dim=64, delta_actions=True,
                 flow_ratio=0.5, dispersive_weight=0.01, depth_noise_std=0.006,
                 pixel_dropout=0.02, view_dropout=0.1, adaptive_gamma=0.5,
                 separate_gripper=False, action_loss_weight=0.0, tcp_position_weight=0.0,
                 tcp_orientation_weight=0.0, gripper_loss_weight=1.0, phase_weight=1.0,
                 tcp_position_scale_m=0.03, tcp_orientation_scale_rad=0.25, use_rgb=False,
                 rgb_color_jitter=0.2, spatial_softmax=False, keypoints=32, rgb_channel_gain_jitter=0.0,
                 state_noise_std=0.0, state_history_dropout=0.0, rgb_backbone="cnn", pretrained_backbone=True,
                 freeze_backbone_stem=False, random_shift_px=0, rgb_cutout_prob=0.0, rgb_blur_prob=0.0):
        super().__init__()
        if (not down_dims or any(d < 8 or d % 8 for d in down_dims)
                or min(horizon, n_obs_steps, n_action_steps) < 1
                or horizon % 2 ** (len(down_dims) - 1) or n_action_steps > horizon):
            raise ValueError("Invalid horizon, action chunk or U-Net widths")
        if time_dim < 4 or time_dim % 2 or min(feature_dim, state_dim) < 1:
            raise ValueError("Invalid feature/time dimensions")
        if not all(0 <= x <= 1 for x in (flow_ratio, pixel_dropout, view_dropout, adaptive_gamma)):
            raise ValueError("Ratios and adaptive_gamma must be in [0,1]")
        if min(dispersive_weight, depth_noise_std, rgb_color_jitter, rgb_channel_gain_jitter, state_noise_std) < 0 or keypoints < 1 or not 0 <= state_history_dropout <= 1 \
                or random_shift_px < 0 or not (0 <= rgb_cutout_prob <= 1 and 0 <= rgb_blur_prob <= 1):
            raise ValueError("Loss/noise weights cannot be negative")
        if min(action_loss_weight, tcp_position_weight, tcp_orientation_weight, gripper_loss_weight) < 0:
            raise ValueError("Auxiliary loss weights cannot be negative")
        if phase_weight < 1 or min(tcp_position_scale_m, tcp_orientation_scale_rad) <= 0:
            raise ValueError("Invalid phase weight or TCP normalization scales")
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.delta_actions, self.flow_ratio = delta_actions, flow_ratio
        self.dispersive_weight, self.adaptive_gamma = dispersive_weight, adaptive_gamma
        self.depth_noise_std, self.pixel_dropout, self.view_dropout = depth_noise_std, pixel_dropout, view_dropout
        self.separate_gripper = separate_gripper
        self.use_rgb, self.rgb_color_jitter = use_rgb, rgb_color_jitter
        self.rgb_channel_gain_jitter, self.state_noise_std = rgb_channel_gain_jitter, state_noise_std
        self.state_history_dropout = state_history_dropout
        self.random_shift_px, self.rgb_cutout_prob, self.rgb_blur_prob = random_shift_px, rgb_cutout_prob, rgb_blur_prob
        self.action_loss_weight, self.tcp_position_weight = action_loss_weight, tcp_position_weight
        self.tcp_orientation_weight, self.gripper_loss_weight = tcp_orientation_weight, gripper_loss_weight
        self.phase_weight = phase_weight
        self.tcp_position_scale_m, self.tcp_orientation_scale_rad = tcp_position_scale_m, tcp_orientation_scale_rad
        self.register_buffer("arm_indices", torch.tensor([i for i in range(16) if i not in (7, 15)]), persistent=False)
        self.register_buffer("gripper_indices", torch.tensor([7, 15]), persistent=False)
        self.flow_dim = 14 if separate_gripper else 16
        self.encoder = DualDepthEncoder(feature_dim, state_dim, use_rgb, spatial_softmax, keypoints, rgb_backbone,
                                       pretrained_backbone, freeze_backbone_stem)
        condition_dim = self.encoder.output_dim * n_obs_steps
        self.velocity = MeanVelocityUNet(condition_dim, down_dims, time_dim, self.flow_dim)
        self.gripper_head = None
        if separate_gripper:
            self.gripper_head = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, horizon * 2))
            nn.init.constant_(self.gripper_head[-1].bias, 2.0)
        self.kinematics = None
        if tcp_position_weight or tcp_orientation_weight:
            from openarm_flow.kinematics import OpenArmKinematics
            self.kinematics = OpenArmKinematics()
        for name in ("state_mean", "action_mean"):
            self.register_buffer(name, torch.zeros(16))
        for name in ("state_scale", "action_scale"):
            self.register_buffer(name, torch.ones(16))
        self.register_buffer("stats_ready", torch.tensor(False))
        arm_mask = torch.ones(16)
        arm_mask[[7, 15]] = 0
        self.register_buffer("arm_mask", arm_mask)

    def set_statistics(self, statistics):
        for name in ("state_mean", "state_scale", "action_mean", "action_scale"):
            value = torch.as_tensor(statistics[name], device=self.state_mean.device)
            if value.shape != (16,) or not torch.isfinite(value).all() or ("scale" in name and (value <= 0).any()):
                raise ValueError(f"Invalid {name}")
            getattr(self, name).copy_(value)
        self.stats_ready.fill_(True)

    def normalize_actions(self, actions, state):
        if self.delta_actions:
            actions = actions - state[:, None] * self.arm_mask
        return (actions - self.action_mean) / self.action_scale

    def unnormalize_actions(self, actions, state):
        actions = actions * self.action_scale + self.action_mean
        return actions + state[:, None] * self.arm_mask if self.delta_actions else actions

    def _grippers(self, condition):
        return .043 * torch.sigmoid(self.gripper_head(condition).reshape(-1, self.horizon, 2))

    def _decode_endpoint(self, condition, state, noise, grippers=None):
        t, r = condition.new_ones(len(condition)), condition.new_zeros(len(condition))
        normalized = noise - self.velocity(noise, t, r, condition)
        if self.separate_gripper:
            normalized = normalized.new_zeros(len(condition), self.horizon, 16).index_copy(-1, self.arm_indices, normalized)
        actions = self.unnormalize_actions(normalized, state)
        if self.separate_gripper:
            actions = actions.index_copy(-1, self.gripper_indices, self._grippers(condition) if grippers is None else grippers)
        return actions

    def _phase_weights(self, batch):
        valid = batch["action_valid"].to(batch["action"].dtype)
        phase = batch.get("action_phase")
        per_arm = valid[..., None].expand(-1, -1, 2)
        if phase is not None:
            per_arm = per_arm * (1 + (self.phase_weight - 1) * phase.to(valid.dtype))
        return per_arm

    @staticmethod
    def _weighted_mean(error, weight):
        return (error * weight).sum() / weight.sum().clamp_min(1)

    def encode(self, obs, augment=False):
        depth, state = obs["depth"], obs["agent_pos"]
        if state.shape[1] != self.n_obs_steps:
            raise ValueError("Observation history length differs from model")
        if augment:
            depth = depth.clone()
            valid = depth[:, :, :, 1:2]
            if self.pixel_dropout:
                valid = valid * (torch.rand_like(valid) >= self.pixel_dropout)
            depth[:, :, :, :1] = (depth[:, :, :, :1] + self.depth_noise_std * torch.randn_like(valid)).clamp(-1, 1) * valid
            depth[:, :, :, 1:2] = valid
            # Drop at most one camera, consistently across observation history.
            drop = torch.rand(len(depth), device=depth.device) < self.view_dropout
            camera = torch.randint(2, (len(depth),), device=depth.device)
            depth[torch.arange(len(depth), device=depth.device)[drop], :, camera[drop]] = 0
        rgb = self._prepare_rgb(obs, augment) if self.use_rgb else None
        if augment and self.random_shift_px:
            depth, rgb = self._random_shift(depth, rgb)
        if augment and self.state_noise_std:
            # Encoder input only (targets unchanged): robustness to the small state errors of closed-loop rollouts.
            noise = self.state_noise_std * torch.randn_like(state)
            noise[..., self.gripper_indices] *= 0.1
            state = state + noise
        if augment and self.state_history_dropout:
            # Hide joint-motion history (keep only the current state) so timing cues such as
            # "when to close" must come from the images, not from how the joints have been moving.
            drop = torch.rand(len(state), device=state.device) < self.state_history_dropout
            state = torch.where(drop[:, None, None], state[:, -1:].expand_as(state), state)
        return self.encoder(depth, (state - self.state_mean) / self.state_scale, rgb)

    def _prepare_rgb(self, obs, augment):
        if "rgb" not in obs:
            raise ValueError("Model uses RGB but the observation has no 'rgb' [B,T,2,3,H,W]")
        rgb = obs["rgb"].float() / 255 if obs["rgb"].dtype == torch.uint8 else obs["rgb"].float()
        if augment:
            b, device = len(rgb), rgb.device
            # Per-sample, per-camera brightness/contrast jitter, constant over the history.
            jitter = self.rgb_color_jitter
            brightness = 1 + jitter * (2 * torch.rand(b, 1, 2, 1, 1, 1, device=device) - 1)
            contrast = 1 + jitter * (2 * torch.rand(b, 1, 2, 1, 1, 1, device=device) - 1)
            mean = rgb.mean((-3, -2, -1), keepdim=True)
            rgb = ((rgb - mean) * contrast + mean) * brightness
            if self.rgb_channel_gain_jitter:  # white-balance shift, per sample and camera
                gain = 1 + self.rgb_channel_gain_jitter * (2 * torch.rand(b, 1, 2, 3, 1, 1, device=device) - 1)
                rgb = rgb * gain
            rgb = rgb.clamp(0, 1)
        rgb = (rgb - .5) / .25
        if augment and self.view_dropout:
            drop = torch.rand(len(rgb), device=rgb.device) < self.view_dropout
            camera = torch.randint(2, (len(rgb),), device=rgb.device)
            rgb[torch.arange(len(rgb), device=rgb.device)[drop], :, camera[drop]] = 0
        if augment and self.rgb_blur_prob:
            rgb = self._random_blur(rgb)
        if augment and self.rgb_cutout_prob:
            rgb = self._random_cutout(rgb)
        return rgb

    def _random_shift(self, depth, rgb):
        """Translate every camera view by a random whole-pixel offset (same for its depth and RGB and for all
        history frames), edges replicated: robustness to small camera/robot misplacement in the image."""
        b, t, v = depth.shape[:3]
        pad = self.random_shift_px
        offsets = torch.randint(-pad, pad + 1, (b, 1, v, 2), device=depth.device).expand(b, t, v, 2).reshape(-1, 2)

        def shift(x):
            n, (h, w) = b * t * v, x.shape[-2:]
            flat = x.reshape(n, -1, h, w)
            theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
            theta[:, 0, 0] = theta[:, 1, 1] = 1
            theta[:, 0, 2] = 2 * offsets[:, 0].to(x.dtype) / w
            theta[:, 1, 2] = 2 * offsets[:, 1].to(x.dtype) / h
            grid = F.affine_grid(theta, flat.shape, align_corners=False)
            return F.grid_sample(flat, grid, mode="nearest", padding_mode="border", align_corners=False).reshape(x.shape)

        return shift(depth), (shift(rgb) if rgb is not None else None)

    def _random_blur(self, rgb):
        b, t, v, c, h, w = rgb.shape
        k = torch.tensor([1., 4., 6., 4., 1.], device=rgb.device, dtype=rgb.dtype)
        kernel = (k[:, None] * k[None]) / 256
        flat = rgb.reshape(-1, 1, h, w)
        blurred = F.conv2d(F.pad(flat, (2, 2, 2, 2), mode="replicate"), kernel[None, None]).reshape(rgb.shape)
        pick = (torch.rand(b, 1, v, 1, 1, 1, device=rgb.device) < self.rgb_blur_prob)
        return torch.where(pick, blurred, rgb)

    def _random_cutout(self, rgb):
        """Gray out one random rectangle (10-30% of each side) of some views, the same for all history frames."""
        b, t, v, c, h, w = rgb.shape
        rgb = rgb.clone()
        pick = torch.rand(b, v) < self.rgb_cutout_prob
        for i, j in pick.nonzero().tolist():
            ch, cw = int(h * (0.1 + 0.2 * torch.rand(()))), int(w * (0.1 + 0.2 * torch.rand(())))
            y, x = int(torch.randint(0, h - ch + 1, ())), int(torch.randint(0, w - cw + 1, ()))
            rgb[i, :, j, :, y:y + ch, x:x + cw] = 0
        return rgb

    def compute_loss(self, batch):
        if not self.stats_ready:
            raise ValueError("Fit training statistics first")
        x = self.normalize_actions(batch["action"], batch["obs"]["agent_pos"][:, -1])
        if self.separate_gripper:
            x = x.index_select(-1, self.arm_indices)
        cond = self.encode(batch["obs"], augment=self.training)
        times = torch.sigmoid(torch.randn(len(x), 2, device=x.device) - 0.4)
        r, t = times.min(dim=1).values, times.max(dim=1).values
        r = torch.where(torch.rand_like(t) < self.flow_ratio, t, r)
        noise = torch.randn_like(x)
        z = (1 - t[:, None, None]) * x + t[:, None, None] * noise
        target = meanflow_target(self.velocity, z, t, r, noise - x, cond)
        prediction, features = self.velocity(z, t, r, cond, return_features=True)
        arm_weight = self._phase_weights(batch)
        # Weight phase supervision for each arm independently; padding is excluded.
        channel_weight = arm_weight.repeat_interleave(7 if self.separate_gripper else 8, dim=-1)
        valid = batch["action_valid"].to(x.dtype)[..., None]
        denominator = (valid.sum((1, 2)) * self.flow_dim).clamp_min(1)
        squared_error = (prediction - target).square()
        mse = (squared_error * valid).sum((1, 2)) / denominator
        weighted_mse = (squared_error * channel_weight).sum((1, 2)) / denominator
        # Do not divide the phase weight back out for an all-approach sample.
        flow = (weighted_mse / (mse.detach() + 1e-3).pow(self.adaptive_gamma)).mean()
        dispersion = dispersive_loss(features) if self.dispersive_weight else flow.new_zeros(())
        loss = flow + self.dispersive_weight * dispersion
        parts = {"meanflow_loss": float(flow.detach()), "dispersive_loss": float(dispersion.detach()),
                 "velocity_mse": float(mse.detach().mean())}
        grippers = self._grippers(cond) if self.separate_gripper else None
        if self.separate_gripper:
            grip_error = F.smooth_l1_loss(grippers / .043, batch["action"][..., [7, 15]].clamp(0, .043) / .043,
                                          reduction="none", beta=.05)
            grip_loss = self._weighted_mean(grip_error, arm_weight)
            loss = loss + self.gripper_loss_weight * grip_loss
            parts["gripper_loss"] = float(grip_loss.detach())
        if self.action_loss_weight or self.kinematics is not None:
            # Train the exact deployed t=1,r=0 endpoint, in addition to the
            # MeanFlow identity. This forward keeps encoder/UNet gradients.
            endpoint = self._decode_endpoint(cond, batch["obs"]["agent_pos"][:, -1], torch.randn_like(x), grippers)
            normalized_endpoint = self.normalize_actions(endpoint, batch["obs"]["agent_pos"][:, -1])[..., self.arm_indices]
            normalized_target = self.normalize_actions(batch["action"], batch["obs"]["agent_pos"][:, -1])[..., self.arm_indices]
            error = F.smooth_l1_loss(normalized_endpoint, normalized_target, reduction="none", beta=.5)
            action_loss = self._weighted_mean(error, arm_weight.repeat_interleave(7, dim=-1))
            loss = loss + self.action_loss_weight * action_loss
            parts["action_loss"] = float(action_loss.detach())
            if self.kinematics is not None:
                pos, rot = self.kinematics(endpoint)
                target_pos, target_rot = self.kinematics(batch["action"], clamp=True)
                pos_error = F.smooth_l1_loss((pos - target_pos) / self.tcp_position_scale_m,
                                            torch.zeros_like(pos), reduction="none").mean(-1)
                # Chordal rotation loss avoids the singular acos gradient at R=R*.
                rot_error = (rot - target_rot).square().mean((-1, -2)) / self.tcp_orientation_scale_rad**2
                position_loss = self._weighted_mean(pos_error, arm_weight)
                orientation_loss = self._weighted_mean(rot_error, arm_weight)
                loss = loss + self.tcp_position_weight * position_loss + self.tcp_orientation_weight * orientation_loss
                parts.update(tcp_position_loss=float(position_loss.detach()), tcp_orientation_loss=float(orientation_loss.detach()))
        return loss, parts

    @torch.no_grad()
    def predict_action(self, obs, *, noise=None, generator=None):
        if not self.stats_ready:
            raise ValueError("Fit training statistics first")
        cond = self.encode(obs)
        shape = (len(cond), self.horizon, self.flow_dim)
        if noise is None:
            noise = torch.randn(shape, device=cond.device, dtype=cond.dtype, generator=generator)
        if noise.shape != shape:
            raise ValueError("Incorrect action noise shape")
        actions = self._decode_endpoint(cond, obs["agent_pos"][:, -1], noise)
        if self.separate_gripper and self.kinematics is not None:
            actions = self.kinematics.clamp(actions)
        return {"action": actions[:, :self.n_action_steps], "action_pred": actions}

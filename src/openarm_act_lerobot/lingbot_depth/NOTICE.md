# LingBot-Depth encoder (vendored)

The RGB-D encoder of [LingBot-Depth](https://github.com/robbyant/lingbot-depth)
(Robbyant, "Masked Depth Modeling for Spatial Perception", arXiv 2601.17895),
Apache License 2.0 (`LICENSE`), itself built on Meta's DINOv2 (Apache 2.0).

Copied from `lingbotvla/models/vla/vision_models/lingbot-depth/mdm/model/` of
[Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2) at commit
`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`. Only the encoder is kept: the
decoder heads, the training utilities and the pretrained-DINOv2 download are left out.

Changes:

- `rgbd_encoder.py` (was `modules_rgbd_encoder.py`): `init_weights`, the
  gradient-checkpointing and SDPA helpers removed; the forward pass is unchanged.
- `dinov2_rgbd/models/vision_transformer.py`: the depth patch embedding is
  `depth_patch_embed`, as the published checkpoint
  `robbyant/lingbot-depth-pretrain-vitl-14` names it. lingbot-vla-v2's
  commit 311a4d2 renamed it `depth_mask_patch_embed`, which that checkpoint
  does not have; loaded with `strict=False`, the layer then kept random
  weights. `openarm_act_lerobot.backbones.LingBotDepthEncoder` accepts either
  name and loads strictly.
- `dinov2_rgbd/models/__init__.py`: the training-time `build_model` removed.

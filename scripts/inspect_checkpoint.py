#!/usr/bin/env python3
"""How much a checkpoint's backbone listens to each input channel (R, G, B,
depth): the L2 norm of its first layer's weights per input channel. With
depth as the 4th channel, its weights start at zero (data.depth_init=zero);
still near zero after training means the model does not use depth. With a
depth branch, the norm of its output projection, which starts at zero too.

    uv run scripts/inspect_checkpoint.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/r1_rgbd
"""

import hydra
from omegaconf import DictConfig

FIRST_LAYERS = ("patch_embeddings.weight", "patch_embeddings.projection.weight", "body.conv1.weight")


@hydra.main(version_base="1.3", config_path="../conf", config_name="inspect_checkpoint")
def main(cfg: DictConfig) -> None:
    from safetensors.torch import load_file

    from openarm_act_lerobot.run import model_dir

    directory = model_dir(cfg.checkpoint)
    weights = load_file(directory / "model.safetensors")
    print(directory)
    first = next(k for k in weights if k.startswith("model.backbone.") and k.endswith(FIRST_LAYERS)
                 and ".depth." not in k)
    norms = [float(weights[first][:, c].norm()) for c in range(weights[first].shape[1])]
    names = ["R", "G", "B", "depth"][: len(norms)]
    print(f"{first}: " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, norms, strict=True)))
    if len(norms) == 4:
        print(f"depth / mean(RGB) = {norms[3] / (sum(norms[:3]) / 3):.4f}")
    projection = "model.backbone.depth.proj.weight"
    if projection in weights:  # depth_as=branch: the depth encoder's output, zero at the start
        print(f"{projection}: norm {float(weights[projection].norm()):.4f} (0 = depth not used)")


if __name__ == "__main__":
    main()

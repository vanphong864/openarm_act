"""Command-line flags for the training scripts, as shorthands for Hydra overrides.

The scripts take Hydra's `key=value`; Hydra refuses `--flags` it does not
know. These are turned into overrides before Hydra reads sys.argv:

    --resnet                  backbone=resnet18 (act_pipeline's)
    --dinov2                  backbone=dinov2_vitb14
    --dinov3                  backbone=dinov3_vitb16
    --backbone NAME           backbone=NAME (any of conf/backbone/), also --backbone=NAME
    --depth-branch            data=openarm_rgbd_branch: raw depth through a small CNN of its own
    --lingbot-depth           the same, with LingBot-Depth as the depth encoder (also at inference)
    --lingbot-depth-distill   distill=lingbot_depth: LingBot-Depth as a training-only teacher

The depth flags add data=openarm_rgbd_branch only when the command line names
no data= of its own (data.depth_as=none with --lingbot-depth-distill: an
RGB-only student). Hydra's own flags (--help, --cfg, -m, ...) pass through;
any other --flag is an error.
"""

from __future__ import annotations

import sys

BRANCH_DATA = "data=openarm_rgbd_branch"
FLAGS: dict[str, list[str]] = {
    "--resnet": ["backbone=resnet18"],
    "--dinov2": ["backbone=dinov2_vitb14"],
    "--dinov3": ["backbone=dinov3_vitb16"],
    "--depth-branch": [BRANCH_DATA],
    "--lingbot-depth": [BRANCH_DATA, "depth_encoder=lingbot_depth"],
    "--lingbot-depth-distill": [BRANCH_DATA, "distill=lingbot_depth"],
}
HYDRA_FLAGS = {
    "--help", "-h", "--hydra-help", "--version", "--cfg", "-c", "--resolve", "--package", "-p", "--run", "-r",
    "--multirun", "-m", "--shell-completion", "-sc", "--config-path", "-cp", "--config-name", "-cn",
    "--config-dir", "-cd", "--experimental-rerun", "--info", "-i",
}


def expand_flags(argv: list[str]) -> list[str]:
    """argv with this module's flags turned into Hydra overrides."""
    out, overrides, i = [argv[0]], [], 1
    while i < len(argv):
        arg = argv[i]
        name = arg.split("=", 1)[0]
        if name == "--backbone":
            if "=" in arg:
                value = arg.split("=", 1)[1]
            elif i + 1 < len(argv):
                i += 1
                value = argv[i]
            else:
                raise SystemExit("--backbone needs a name, e.g. --backbone dinov2_vits14")
            overrides.append(f"backbone={value}")
        elif arg in FLAGS:
            overrides += FLAGS[arg]
        elif arg.startswith("-") and name not in HYDRA_FLAGS:
            known = ", ".join([*FLAGS, "--backbone NAME"])
            raise SystemExit(f"unknown flag {arg}; this script knows {known}, and Hydra's key=value overrides")
        else:
            out.append(arg)
        i += 1

    if "--lingbot-depth" in argv and "--lingbot-depth-distill" in argv:
        raise SystemExit("--lingbot-depth and --lingbot-depth-distill: LingBot-Depth is either the depth encoder "
                         "or the teacher; pick one")
    given = {arg.split("=", 1)[0] for arg in out[1:] if "=" in arg and not arg.startswith("-")}
    seen = set()
    for override in overrides:
        key = override.split("=", 1)[0]
        if key in given:  # what the command line says itself wins
            continue
        if key in seen:
            if override not in out:
                raise SystemExit(f"flags disagree about {key}: {override} and an earlier one")
            continue
        seen.add(key)
        out.append(override)
    return out


def apply() -> None:
    """For a script's `if __name__ == "__main__":`, before main()."""
    sys.argv = expand_flags(sys.argv)

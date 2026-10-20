#!/usr/bin/env python3
"""Convert two fixed raw depth views to a MeanFlow dataset; source HDF5 is read-only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openarm_flow.depth_views import DepthConfig
from openarm_flow.meanflow_data import convert_depth
from openarm_flow.rgb_views import RGBConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("manifests/episodes.yaml"))
    ap.add_argument("--cameras", nargs=2, default=["chest_depth", "front_depth"])
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--no-rgb", action="store_true", help="Depth only: skip storing the RGB views (data/rgb)")
    ap.add_argument("--rgb-cameras", nargs=2, default=["chest_rgb", "front_rgb"])
    ap.add_argument("--crop", nargs=5, action="append", metavar=("CAMERA", "Y0", "Y1", "X0", "X1"), default=[],
                    help="Fixed region of interest of a camera (recorded pixels), shared by its depth and RGB; "
                         "e.g. --crop chest 80 480 0 640 --crop front 0 336 0 640")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--action-source", choices=["commands", "next_qpos", "qpos"], default="commands")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-invalid", action="store_true", help="Record rejected episodes and their reasons in metadata")
    ap.add_argument("--object-features", action="store_true",
                    help="Add object-centric features (tin/pad pose from the first frame + FK), needs --rgb data")
    ap.add_argument("--all-train", action="store_true",
                    help="Every episode is training data (extra data such as simulated demos; no validation split)")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--episodes", nargs="+", help="Explicit episode filenames for a smoke test")
    args = ap.parse_args()
    crops = {c: [int(v) for v in box] for c, *box in args.crop}
    objects = None
    if args.object_features:
        from openarm_flow.object_features import ObjectConfig
        objects = ObjectConfig.from_measurements()
    cfg = DepthConfig(cameras=tuple(args.cameras), height=args.height, width=args.width, depth_scale=args.depth_scale,
                      crops=crops)
    rgb = None if args.no_rgb else RGBConfig(cameras=tuple(args.rgb_cameras), height=args.height, width=args.width, crops=crops)
    metadata = convert_depth(args.raw, args.output, cfg, manifest=args.manifest, action_source=args.action_source,
                             val_fraction=args.val_fraction, seed=args.seed, max_episodes=args.max_episodes,
                             max_frames=args.max_frames, skip_invalid=args.skip_invalid, episode_names=args.episodes, rgb=rgb,
                             all_train=args.all_train, objects=objects)
    print(f"Saved {len(metadata['episodes'])} episodes; skipped {len(metadata['skipped'])}; {args.output}")


if __name__ == "__main__":
    main()

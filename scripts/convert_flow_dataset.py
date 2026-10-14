#!/usr/bin/env python3
"""Convert OpenArm HDF5 to FlowPolicy Zarr using explicit camera calibration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openarm_flow.data import convert
from openarm_flow.pointcloud import Camera, PointCloudConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--camera", type=Path, required=True)
    ap.add_argument("--allow-approximate-camera", action="store_true")
    ap.add_argument("--manifest", type=Path, default=Path("manifests/episodes.yaml"))
    ap.add_argument("--action-source", choices=["commands", "next_qpos", "qpos"], default="commands")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--num-points", type=int, default=512)
    ap.add_argument("--min-valid", type=int, default=32)
    ap.add_argument("--xyz-min", type=float, nargs=3)
    ap.add_argument("--xyz-max", type=float, nargs=3)
    ap.add_argument("--use-color", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--max-frames", type=int, help="Debug only: first N labelled frames of each episode")
    args = ap.parse_args()
    for name in ["max_episodes", "max_frames"]:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            ap.error(f"{name} must be positive")
    camera = Camera.load(args.camera)
    if "approximate" in camera.provenance.lower() and not args.allow_approximate_camera:
        ap.error("Estimated calibration: pass --allow-approximate-camera explicitly for an experiment")
    cloud = PointCloudConfig(num_points=args.num_points, min_valid=args.min_valid,
                             use_color=args.use_color, xyz_min=args.xyz_min, xyz_max=args.xyz_max)
    meta = convert(args.raw, args.output, camera, cloud, manifest=args.manifest, fps=args.fps,
                   action_source=args.action_source, val_fraction=args.val_fraction, seed=args.seed,
                   max_episodes=args.max_episodes, max_frames=args.max_frames)
    print(f"Saved {len(meta['episodes'])} episodes to {args.output}; calibration: {camera.provenance}")


if __name__ == "__main__":
    main()

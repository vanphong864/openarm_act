#!/usr/bin/env python3
"""Prepare measured landmark inputs or solve an independent camera-pose proposal."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw

from openarm_data.episode import EpisodeReader
from openarm_flow.scene_calibration import solve_measured_landmarks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--episode", type=Path, required=True)
    prepare.add_argument("--frame", type=int, default=64)
    prepare.add_argument("--out", type=Path, required=True)
    solve = sub.add_parser("solve")
    solve.add_argument("--measurements", type=Path, required=True)
    solve.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "prepare":
        if args.out.exists():
            raise FileExistsError("Choose a new directory; do not overwrite measured points")
        with EpisodeReader(args.episode) as reader:
            if not 0 <= args.frame < len(reader):
                raise ValueError("Frame outside episode")
            rgb, depth = reader.rgb(args.frame), reader.depth(args.frame)
        args.out.mkdir(parents=True)
        Image.fromarray(rgb).save(args.out / "reference_rgb.png")
        Image.fromarray(depth).save(args.out / "reference_depth_mm.png")
        spec = {"schema_version": 1, "world_frame": "robot_base_x_forward_y_left_z_up_m",
                "image": {"path": str((args.out / "reference_rgb.png").resolve()), "width": rgb.shape[1],
                          "height": rgb.shape[0], "source_episode": str(args.episode.resolve()), "source_frame": args.frame},
                "intrinsics": {"K": [[None, 0, None], [0, None, None], [0, 0, 1]], "distortion": [],
                               "pixel_grid": "recorded_rgb", "stream": "color", "provenance": None},
                "measurements": {"source": "physical_measurement", "description": None},
                "points": [{"id": f"measured_point_{i + 1}", "role": "fit" if i < 6 else "check",
                            "world_xyz_m": [None, None, None], "pixel_uv": [None, None]} for i in range(8)]}
        (args.out / "measured_landmarks.json").write_text(json.dumps(spec, indent=2))
        print(f"Prepared {args.out}; fill factory/board RGB intrinsics, measured world XYZ and observed UV")
        return
    spec = json.loads(args.measurements.read_text())
    report = solve_measured_landmarks(spec)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "camera_pose_proposal.json").write_text(json.dumps(report, indent=2))
    image_path = Path(spec["image"]["path"])
    if image_path.is_file():
        image = Image.open(image_path).convert("RGB")
        if image.size != (spec["image"]["width"], spec["image"]["height"]):
            raise ValueError("Reference image dimensions differ from calibration")
        draw = ImageDraw.Draw(image)
        for item in report["landmarks"]:
            u, v = item["observed_uv"]
            x, y = item["projected_uv"]
            draw.ellipse((u-4, v-4, u+4, v+4), outline="lime", width=2)
            draw.line((u, v, x, y), fill="red", width=2)
            draw.text((u+5, v), f"{item['id']} {item['error_px']:.1f}px", fill="yellow")
        image.save(args.out / "landmark_reprojection.png")
    print(json.dumps({k: report[k] for k in ("status", "fit_rmse_px", "heldout_rmse_px", "heldout_max_px", "applied_to_scene")}, indent=2))
    if report["status"] != "proposal_passes_pixel_checks":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

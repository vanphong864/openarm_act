#!/usr/bin/env python3
"""Run the imported OpenArm workspace with a viewer or offscreen preview."""

import argparse
import os
from pathlib import Path
import sys
import time


def clean_snap_gui_environment():
    """Use host GTK libraries/schemas when launched from a Snap editor terminal."""
    data_dirs = os.environ.get("XDG_DATA_DIRS")
    if data_dirs and any("/snap/" in p for p in data_dirs.split(":")):
        os.environ["XDG_DATA_DIRS"] = ":".join(
            p for p in data_dirs.split(":") if "/snap/" not in p
        ) or "/usr/local/share:/usr/share"
    for key in (
        "GSETTINGS_SCHEMA_DIR", "GTK_EXE_PREFIX", "GTK_PATH",
        "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR",
        "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR",
    ):
        value = os.environ.get(key, "")
        if "/snap/" in value:
            os.environ.pop(key, None)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true", help="Save camera and overview images instead of opening a window")
    ap.add_argument("--seconds", type=float, default=2.0, help="Headless simulation duration")
    ap.add_argument("--out", type=Path, default=root / "outputs/simulation")
    ap.add_argument("--config", type=Path, default=root / "simulation/config/scene_openarm.yaml")
    ap.add_argument("--scene", type=Path, default=root / "simulation/scenes/openarm.xml")
    args = ap.parse_args()
    if args.seconds < 0:
        ap.error("--seconds must be nonnegative")
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        clean_snap_gui_environment()
    sys.path.insert(0, str(root / "simulation"))
    import mujoco
    from openarm_sim.runner import Sim
    from openarm_sim.scene import load_config, load_scene

    cfg = load_config(args.config)
    model, _ = load_scene(args.scene)
    sim = Sim(model, cfg)
    sim.reset()
    action = sim.qpos16().copy()
    print(f"Loaded {args.scene}: {model.nq} qpos, {model.nu} actuators, {sim.fps} Hz")
    if args.headless:
        from PIL import Image
        from openarm_sim.render import OffscreenRenderer, save_depth
        from openarm_sim.sensor import ChestCamera

        for _ in range(round(args.seconds * sim.fps)):
            sim.step(action)
        args.out.mkdir(parents=True, exist_ok=True)
        cameras = [("chest", cfg["camera"], True)] + [
            (c["name"], c, False) for c in cfg.get("additional_cameras") or []
        ]
        for name, camera_cfg, use_background in cameras:
            camera = ChestCamera(model, camera_cfg["color"]["name"], out_size=(640, 480),
                                 use_background=use_background,
                                 native_size=tuple(camera_cfg["color"]["resolution"]))
            try:
                rgb, depth = camera.observe(sim.d)
                Image.fromarray(rgb).save(args.out / f"{name}_rgb.png")
                save_depth(depth.astype(float) / 1000, str(args.out / f"{name}_depth_mm.png"),
                           str(args.out / f"{name}_depth_vis.png"), max_m=1.2)
            finally:
                camera.close()
        overview = mujoco.MjvCamera()
        overview.lookat[:] = [0.2, 0, 0.3]
        overview.distance, overview.azimuth, overview.elevation = 2.0, 140, -20
        with OffscreenRenderer(model, 960, 720) as renderer:
            Image.fromarray(renderer.rgb(sim.d, overview)).save(args.out / "overview.png")
        print(f"Saved RGB, depth (uint16 mm) for {', '.join(name for name, _, _ in cameras)} "
              f"and overview to {args.out}")
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, sim.d) as viewer:
            while viewer.is_running():
                start = time.monotonic()
                sim.step(action)
                viewer.sync()
                time.sleep(max(0, 1 / sim.fps - (time.monotonic() - start)))


if __name__ == "__main__":
    main()

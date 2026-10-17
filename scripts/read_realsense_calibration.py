#!/usr/bin/env python3
"""Read factory profiles only. No streams started and no robot commands sent."""

import argparse
import json
from pathlib import Path


def main():
    import pyrealsense2 as rs

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"source": "RealSense SDK device factory profiles", "devices": [],
              "note": "Profiles are NOT proof of the stream/alignment/resize used in an old HDF5 episode"}
    try:
        context = rs.context()
        for device in context.query_devices():
            item = {"name": device.get_info(rs.camera_info.name), "serial": device.get_info(rs.camera_info.serial_number),
                    "firmware": device.get_info(rs.camera_info.firmware_version), "profiles": []}
            for sensor in device.query_sensors():
                if sensor.is_depth_sensor():
                    item["native_depth_scale_m"] = sensor.as_depth_sensor().get_depth_scale()
                for profile in sensor.get_stream_profiles():
                    if not profile.is_video_stream_profile() or profile.stream_type() not in (rs.stream.color, rs.stream.depth):
                        continue
                    video = profile.as_video_stream_profile()
                    intrinsic = video.get_intrinsics()
                    item["profiles"].append({"stream": str(profile.stream_type()), "format": str(profile.format()),
                                             "fps": profile.fps(), "width": intrinsic.width, "height": intrinsic.height,
                                             "K": [[intrinsic.fx, 0, intrinsic.ppx], [0, intrinsic.fy, intrinsic.ppy], [0, 0, 1]],
                                             "distortion_model": str(intrinsic.model), "distortion_coefficients": list(intrinsic.coeffs)})
            report["devices"].append(item)
        report["status"] = "read_factory_profiles" if report["devices"] else "no_connected_realsense"
    except RuntimeError as exc:
        report.update(status="usb_access_error", error=str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "devices": len(report["devices"]), "output": str(args.out)}))
    if report["status"] != "read_factory_profiles":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

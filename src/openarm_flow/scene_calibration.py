"""Independent landmark calibration. Never derives world points from model/FK.

OpenCV camera axes: right, down, forward. World axes: robot-base x forward,
y left, z up, metres. A proposal is returned; no simulator/checkpoint is edited.
"""

import numpy as np


def solve_measured_landmarks(spec, *, check_rmse_limit_px=2.0, check_max_limit_px=5.0):
    import cv2

    if spec.get("world_frame") != "robot_base_x_forward_y_left_z_up_m":
        raise ValueError("World coordinates must use the declared robot-base frame in metres")
    measurements = spec.get("measurements", {})
    if measurements.get("source") != "physical_measurement" or not measurements.get("description"):
        raise ValueError("World points require documented physical measurements, not scene/FK estimates")
    intrinsic = spec["intrinsics"]
    if not intrinsic.get("provenance") or intrinsic.get("pixel_grid") != "recorded_rgb":
        raise ValueError("Document intrinsics for the actual recorded RGB pixel grid")
    k = np.asarray(intrinsic["K"], dtype=float)
    distortion = np.asarray(intrinsic.get("distortion", []), dtype=float).reshape(-1)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.isfinite(distortion).all()
            or len(distortion) not in (0, 4, 5, 8, 12, 14)):
        raise ValueError("Invalid camera intrinsics/distortion")
    points = spec["points"]
    ids = [p["id"] for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError("Landmark IDs must be unique")
    if any(p["role"] not in ("fit", "check") for p in points):
        raise ValueError("Each point needs role fit or check")
    xyz = np.asarray([p["world_xyz_m"] for p in points], dtype=float)
    uv = np.asarray([p["pixel_uv"] for p in points], dtype=float)
    if xyz.shape != (len(points), 3) or uv.shape != (len(points), 2) or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Fill finite measured XYZ and observed image UV for all landmarks")
    size = spec["image"]
    width, height = int(size["width"]), int(size["height"])
    if width < 1 or height < 1 or np.any(uv < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] >= height):
        raise ValueError("Landmark pixels outside recorded image")
    fit = np.array([p["role"] == "fit" for p in points])
    if fit.sum() < 6 or (~fit).sum() < 2:
        raise ValueError("Need at least 6 fit points and 2 independent check points")
    spread = np.linalg.svd(xyz[fit] - xyz[fit].mean(0), compute_uv=False)
    if spread[-1] < 1e-4:
        raise ValueError("Fit points must include different heights; planar-only pose ambiguity is not accepted here")
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        xyz[fit], uv[fit], k, distortion if len(distortion) else None,
        iterationsCount=500, reprojectionError=3.0, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError("Pose fit failed or fewer than 6 inliers")
    inliers = inliers[:, 0]
    rotation, translation = cv2.solvePnPRefineLM(xyz[fit][inliers], uv[fit][inliers], k,
                                                distortion if len(distortion) else None, rotation, translation)
    rcw = cv2.Rodrigues(rotation)[0]
    translation = translation[:, 0]
    camera_points = xyz @ rcw.T + translation
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Fitted pose puts landmarks behind the camera")
    projected = cv2.projectPoints(xyz, rotation, translation, k, distortion if len(distortion) else None)[0][:, 0]
    errors = np.linalg.norm(projected - uv, axis=1)
    camera_position = -rcw.T @ translation
    r_world_cv = rcw.T
    r_world_gl = r_world_cv @ np.diag([1., -1., -1.])
    heldout_rmse = float(np.sqrt(np.mean(errors[~fit]**2)))
    heldout_max = float(errors[~fit].max())
    quality_pass = heldout_rmse <= check_rmse_limit_px and heldout_max <= check_max_limit_px
    return {"status": "proposal_passes_pixel_checks" if quality_pass else "rejected_by_heldout_checks",
            "applied_to_scene": False, "world_frame": spec["world_frame"],
            "world_measurement_source": measurements,
            "image": size, "intrinsics": intrinsic,
            "camera_position_world_m": camera_position.tolist(),
            "camera_rotation_world_from_opencv": r_world_cv.tolist(),
            "camera_rotation_world_from_mujoco_camera": r_world_gl.tolist(),
            "world_to_opencv_rvec": rotation.reshape(3).tolist(),
            "world_to_opencv_translation_m": translation.tolist(),
            "fit_inliers": len(inliers), "fit_rmse_px": float(np.sqrt(np.mean(errors[fit]**2))),
            "heldout_rmse_px": heldout_rmse, "heldout_max_px": heldout_max,
            "quality_limits_px": {"heldout_rmse": check_rmse_limit_px, "heldout_max": check_max_limit_px},
            "distortion_requires_matching_render_or_rectification": bool(np.any(np.abs(distortion) > 1e-8)),
            "landmarks": [{"id": p["id"], "role": p["role"], "observed_uv": uv[i].tolist(),
                           "projected_uv": projected[i].tolist(), "error_px": float(errors[i])} for i, p in enumerate(points)],
            "limitations": ["Pixel checks do not quantify physical ruler, robot-frame or TCP measurement uncertainty",
                            "A calibration captured after moving the camera cannot automatically calibrate old episodes",
                            "Checkpoint preprocessing is unchanged; do not replace training intrinsics silently"]}

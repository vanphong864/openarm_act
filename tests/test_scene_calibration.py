"""Synthetic geometry checks; these tests never produce real calibration data."""

import copy

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from openarm_flow.scene_calibration import solve_measured_landmarks


@pytest.fixture
def synthetic_landmarks():
    # Six fit points span multiple heights; two entirely separate check points.
    xyz = np.array([[-.2, -.15, 0], [.2, -.15, 0], [.2, .15, 0], [-.2, .15, 0],
                    [-.12, -.08, .12], [.12, .08, .10], [.02, -.05, .03], [-.07, .04, .08]], dtype=float)
    k = np.array([[580., 0, 319.5], [0, 600., 239.5], [0, 0, 1]])
    rotation, translation = np.array([.15, -.08, .04]), np.array([.01, -.02, .9])
    uv = cv2.projectPoints(xyz, rotation, translation, k, None)[0][:, 0]
    spec = {"world_frame": "robot_base_x_forward_y_left_z_up_m", "image": {"width": 640, "height": 480},
            "measurements": {"source": "physical_measurement", "description": "SYNTHETIC UNIT TEST ONLY"},
            "intrinsics": {"K": k.tolist(), "distortion": [], "pixel_grid": "recorded_rgb", "provenance": "SYNTHETIC UNIT TEST ONLY"},
            "points": [{"id": str(i), "role": "fit" if i < 6 else "check", "world_xyz_m": x.tolist(), "pixel_uv": u.tolist()}
                       for i, (x, u) in enumerate(zip(xyz, uv, strict=True))]}
    return spec, rotation, translation


def test_known_pose_and_coordinate_conventions(synthetic_landmarks):
    spec, rotation, translation = synthetic_landmarks
    result = solve_measured_landmarks(spec)
    rcw = cv2.Rodrigues(rotation)[0]
    assert result["status"] == "proposal_passes_pixel_checks"
    assert result["applied_to_scene"] is False
    assert result["heldout_rmse_px"] < 1e-4
    np.testing.assert_allclose(result["camera_position_world_m"], -rcw.T @ translation, atol=1e-5)
    np.testing.assert_allclose(result["camera_rotation_world_from_mujoco_camera"], rcw.T @ np.diag([1, -1, -1]), atol=1e-5)


def test_heldout_points_are_not_used_to_fit(synthetic_landmarks):
    spec, _, _ = synthetic_landmarks
    changed = copy.deepcopy(spec)
    changed["points"][-1]["pixel_uv"][0] += 20
    original = solve_measured_landmarks(spec)
    result = solve_measured_landmarks(changed)
    np.testing.assert_allclose(result["camera_position_world_m"], original["camera_position_world_m"], atol=1e-8)
    assert result["status"] == "rejected_by_heldout_checks"
    assert result["heldout_max_px"] > 19


def test_reject_model_generated_landmarks(synthetic_landmarks):
    spec, _, _ = synthetic_landmarks
    spec["measurements"]["source"] = "scene_or_fk"
    with pytest.raises(ValueError, match="physical measurements"):
        solve_measured_landmarks(spec)


def test_reject_planar_or_unfilled_landmarks(synthetic_landmarks):
    spec, _, _ = synthetic_landmarks
    planar = copy.deepcopy(spec)
    for p in planar["points"]:
        p["world_xyz_m"][2] = 0
    with pytest.raises(ValueError, match="different heights"):
        solve_measured_landmarks(planar)
    spec["points"][0]["world_xyz_m"][0] = None
    with pytest.raises(ValueError, match="finite"):
        solve_measured_landmarks(spec)


def test_missing_check_points_is_not_independent_validation(synthetic_landmarks):
    spec, _, _ = synthetic_landmarks
    for p in spec["points"]:
        p["role"] = "fit"
    with pytest.raises(ValueError, match="independent check"):
        solve_measured_landmarks(spec)

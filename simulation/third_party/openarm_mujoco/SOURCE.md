# Nguồn

- Repo: https://github.com/enactic/openarm_mujoco
- Commit: `e5b9e50` ("Use the managed MuJoCo viewer for real-time simulation (#77)")
- License: Apache-2.0 (xem `LICENSE` trong thư mục này), Copyright 2025 Enactic, Inc.
- Copy nguyên trạng, không sửa:
  - `v1/` (openarm_bimanual.xml, openarm.xml, scene.xml, meshes/)
  - `v0.3/meshes/d435.stl` (vỏ Intel RealSense D435; mặt kính ở z = 0 hướng +z, trên = +y, đơn vị m)

Scene của project được sinh từ `v1/openarm_bimanual.xml` bằng `scripts/build_scene.py`; file gốc không bị thay đổi.

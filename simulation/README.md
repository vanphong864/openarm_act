# OpenArm MuJoCo workspace

> **Dataset 2026-10-13 (hai camera chest + front):** dùng scene đo đạc
> `config/scene_20261013.yaml` → `scenes/openarm_20261013.xml`. Camera, bộ chấp hành, động học robot
> và cảnh đều đo từ dữ liệu, camera ghép nền thật. Xem [docs/SIM_20261013.md](../docs/SIM_20261013.md).
> Phần dưới mô tả workspace cũ (số đo tay, một camera ngực).

Phần mô phỏng tối thiểu lấy từ `/home/space/VR/OpenArm_Sim`, commit
`11025df`. Giữ cấu hình workspace, camera ngực D435i, robot hai tay,
hộp kê, ô đích, vật Mentos và servo của nguồn. Mặc định dùng
`config/scene_openarm.yaml` kế thừa `config/scene.yaml`.

Scene và các mesh nằm trong repo này, không cần checkout `OpenArm_Sim`.
Chỉ lấy các mesh được robot tham chiếu và mesh camera D435. Giấy phép
robot nằm tại `third_party/openarm_mujoco/LICENSE`; thông tin nguồn tại
`third_party/openarm_mujoco/SOURCE.md`.

Từ thư mục gốc `openarm_act`, cài thêm dependency:

```bash
uv pip install --python .venv/bin/python -r simulation/requirements.txt
```

Mở viewer tương tác (cần màn hình):

```bash
.venv/bin/python scripts/simulate.py
```

Chạy vật lý hai giây và xuất ảnh, phù hợp với RunPod:

```bash
.venv/bin/python scripts/simulate.py --headless
```

Ảnh được ghi vào `outputs/simulation/`: `overview.png`, `chest_rgb.png`,
`chest_depth_mm.png` (uint16 mm, 0 là pixel không hợp lệ),
`chest_depth_vis.png`. RGB và depth có kích thước 640×480.
Script giữ lệnh khớp ở tư thế home để kiểm tra workspace.
Script này chỉ kiểm tra scene, không điều khiển robot thật.

Khi thay đổi kích thước hoặc camera trong YAML, sinh lại scene:

```bash
.venv/bin/python simulation/scripts/build_scene.py
```

## Thêm camera depth thứ hai

`config/two_cameras.example.yaml` thêm một D435 ảo với `side_color` và
`side_depth`. Pose mẫu chỉ để thử render; đo vị trí và hướng camera thật rồi
sửa `parent_body`, `pos`, `yaw_deg`, `pitch_deg`, `roll_deg`. Tọa độ `pos`
tính theo body cha; `world` dùng hệ thế giới (x trước, y trái, z lên).
Thay `fovy_deg`/`resolution` theo calibration của camera thứ hai. Build và
render bằng cùng một cặp config/scene:

```bash
.venv/bin/python simulation/scripts/build_scene.py --config simulation/config/two_cameras.example.yaml --out simulation/scenes/openarm_two_cameras.xml
.venv/bin/python scripts/simulate.py --headless --seconds 0 --config simulation/config/two_cameras.example.yaml --scene simulation/scenes/openarm_two_cameras.xml --out outputs/two_cameras
```

Lệnh trên xuất `chest_rgb.png`, `chest_depth_mm.png`, `side_rgb.png`,
`side_depth_mm.png` cùng ảnh depth trực quan. Ảnh depth đầu ra là uint16 mm,
0 là điểm không hợp lệ. Renderer dùng ảnh màu của từng camera để tạo depth đã
căn lên khung màu 640×480. Nếu luồng thu thật là depth nguyên bản 848×480
hoặc khung đã crop/resize khác, cần chỉnh bước căn chỉnh và intrinsics trước
khi dùng ảnh mô phỏng cho policy.

Các converter/policy hiện đọc `chest` và `chest_depth`; để huấn luyện/chạy
với hai góc nhìn còn cần ánh xạ key HDF5 của camera mới vào input model, và
đảm bảo thứ tự, độ phân giải, intrinsics/extrinsics khớp dữ liệu thật.

Kiểm tra hình học và render các góc camera:

```bash
.venv/bin/python simulation/scripts/check_scene.py --out outputs/simulation_check
```

Workspace giữ các số đo từ repo nguồn; cấu hình nguồn vẫn ghi nhận sai lệch
động lực học và hình học so với robot thật. Ảnh nền thật
`data/background_chest.png` không có trong repo nguồn tại thời điểm sao chép;
render dùng nền MuJoCo. Có thể bổ sung ảnh vào `simulation/data/` và build lại.
`check_scene.py` báo tâm hộp và tâm ô đích ngoài bán kính với ước lượng
của tay (ô đích thiếu khoảng 6.4 cm), trả exit code 1 dù load và render
thành công. Đây là kết quả của cấu hình nguồn; cần cân nhắc vị trí thả
ở mép ô khi thử bài gắp đặt.

Mã `openarm_sim` gồm scene builder, renderer, sensor và các dependency để
khởi tạo `Sim`, đọc qpos 16 chiều, nhận joint target 16 chiều tại 50 Hz.
Thứ tự: trái (7 khớp + gripper), phải (7 khớp + gripper).
ROS 2, dataset và các công cụ thu demo của nguồn không được
đưa vào bản mô phỏng này.

## Rollout FlowPolicy

```bash
.venv/bin/python scripts/rollout_mujoco.py --policy flowpolicy --checkpoint releases/flow_rgbd_v1 --seconds 4
```

Thêm `--view` để mở viewer. Kết quả ghi tại `outputs/mujoco_policy/`:
GIF, ảnh cuối, trajectory và result.json. Runner cập nhật observation mỗi tick,
chỉ dùng release FlowPolicy một camera. MeanFlow hai camera cần adapter camera
riêng; xem [hướng dẫn MeanFlow v2](../docs/MEANFLOW_V2.md).

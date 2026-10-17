# Calibration độc lập và sai số TCP pha tiếp cận

Không dùng action policy, qpos lúc gắp hoặc một lần replay thành công để xác định
tọa độ vật/camera thật. Những phép dịch vật từ TCP trong chẩn đoán trước chỉ là
thử độ nhạy, không phải calibration đo độc lập.

## Trạng thái số đo hiện tại

- `conf/flow_camera.d435i_049322070569.yaml` là intrinsics **native depth** 640×480,
  không tự động là intrinsics của RGB hoặc depth đã align lên RGB.
- Episode `20261008_153152` không lưu intrinsics, extrinsics hay thao tác align/resize.
- `OpenArm_Sim/config/camera_landmarks.json` ghi rõ world coordinates lấy từ
  `scene.yaml`; vì vậy không được dùng làm kiểm chứng world coordinates độc lập.
- Kích thước vật Mentos trong scene đang ghi là ước lượng.
- Khi kiểm tra USB toàn máy trong phiên này: không có RealSense được nhận.
- Chưa thay camera pose, kích thước vật hay weight checkpoint bằng các số ước lượng mới.

## Thu số đo cần thiết

1. Xác nhận camera/cảnh còn nguyên vị trí như lúc thu dataset. Nếu đã dịch chuyển,
   calibration hiện tại không tự động áp dụng được cho episode cũ.
2. Cắm camera ngực và đọc factory profiles (không khởi động stream, không gửi lệnh robot):

   ```bash
   .venv/bin/python scripts/read_realsense_calibration.py \
     --out outputs/flow_independent_calibration/realsense_profiles.json
   ```

3. Xác định đúng profile RGB lúc ghi và phép resize/crop, depth là native hay align.
   Không lấy K của depth để thay cho K của RGB. Chỉ biết ảnh là 640×480 chưa đủ
   để suy ra K; cần biết stream gốc và biến đổi đã áp dụng.
4. Đo bằng thước hoặc marker đã biết vị trí: các mốc cố định trên hộp kê, ô đích,
   và ít nhất hai mốc ở độ cao khác. Ghi XYZ theo gốc chân đế, **x trước, y trái,
   z lên, mét**; lấy UV trực tiếp trên ảnh episode. Công cụ hiện yêu cầu ít nhất
   6 điểm fit không đồng phẳng và 2 điểm check không dùng lúc fit.
5. Đo kích thước vật thật và vị trí/yaw ban đầu độc lập, không lấy từ qpos gắp.
   Lưu độ không chắc của phép đo. Muốn đo TCP thực độc lập còn cần marker trên
   đầu kẹp với offset marker→TCP được đo, hoặc một thiết bị đo bên ngoài.

Chuẩn bị ảnh và mẫu điền (không tự điền tọa độ world từ scene):

```bash
.venv/bin/python scripts/calibrate_flow_scene.py prepare \
  --episode data/raw/data_openarm_8_10/episode_20261008_153152.hdf5 \
  --frame 64 --out outputs/calibration_new_reference
```

Điền `measured_landmarks.json`, bao gồm K, distortion đúng grid ảnh, nguồn số đo,
world XYZ và image UV, rồi chạy:

```bash
.venv/bin/python scripts/calibrate_flow_scene.py solve \
  --measurements outputs/calibration_new_reference/measured_landmarks.json \
  --out outputs/calibration_proposal
```

Pose dùng [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html),
RANSAC và refine LM trên fit points. Hai điểm check chỉ đánh giá, không tham gia fit.
Giới hạn kiểm tra mặc định: check RMSE ≤2 px và max ≤5 px. Đây là điều kiện số,
không bảo đảm chính xác vật lý nếu thước, K, gốc tọa độ hoặc pixel annotations sai.
Công cụ xuất proposal và ảnh reprojection, không tự ghi đè scene hay release.
Không chấp nhận các điểm chỉ nằm trên một mặt phẳng trong quy trình bảo thủ này.

MuJoCo camera dùng x phải, y lên, z ngược hướng nhìn; OpenCV dùng x phải, y xuống,
z theo hướng nhìn. Proposal lưu rõ hai rotation để tránh nhầm hệ trục. Nếu ảnh có
distortion, render/rectification phải mô hình hóa tương ứng trước khi so policy.
Renderer cần dùng camera vật lý thật; checkpoint vẫn giữ preprocessing đã train,
không được âm thầm sửa K trong release để làm số liệu đẹp hơn.

## Sai số đầu kẹp theo pha, đã chạy trên dữ liệu hiện có

```bash
.venv/bin/python scripts/measure_flow_tcp.py
```

Script dùng output của `diagnose_flow_episode.py` và rollout vòng kín đã lưu.
Pha được xác định từ demonstration command, không từ model: lần đầu lệnh gripper
phải xuống dưới 30 mm là 5.04 s tính từ frame đầu đã trim; **tiếp cận cuối là
4.04–5.04 s**, 50 target samples. Padding loại khỏi thống kê.

Tách ba loại tham chiếu:

- Sai số dự đoán TCP: FK(action model) so với FK(action demonstration), cùng timestamp;
  cả hai được clip theo giới hạn controller. Không so lệnh trực tiếp với state đo
  vì lệnh có thể đi trước state do servo trễ.
- Sai số bám demonstration: FK(qpos simulator) so với FK(qpos đo thật), cùng timestamp.
- Sai số trajectory vòng kín: FK(qpos policy rollout) so với FK(qpos thật), cùng đồng hồ;
  bao gồm khác biệt tiến độ tác vụ, không thể coi tất cả là sai số hình học thuần.

Tất cả FK dùng cùng URDF. Đây là proxy từ encoder/qpos, **không phải ground truth
TCP đo độc lập**, không kiểm chứng được sai số chiều dài link, lắp đặt hoặc offset TCP.

Số liệu tại `outputs/flow_tcp_approach/report.json`, CSV từng frame và `SUMMARY.md`:

| Phép đo trong tiếp cận cuối | TCP RMSE |
|---|---:|
| Model trên observation thật, seed 42 | 28.64 mm |
| Model trên observation thật, seed 43 | 25.83 mm |
| Model trên observation thật, seed 44 | 21.24 mm |
| Bám lệnh demonstration trong scene mặc định | 9.87 mm |

Baseline giữ nguyên qpos có TCP command RMSE 111.30 mm trong cửa sổ này.
Không cộng/trừ trực tiếp các RMSE để phân rã nguyên nhân; chúng dùng tham chiếu
khác nhau, và chưa có calibration/TCP ground truth độc lập.

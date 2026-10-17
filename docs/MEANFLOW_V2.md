# MeanFlow v2 cho OpenArm V1 — hai camera depth, dữ liệu hiện có

V2 bổ sung loss action/TCP, nhánh gripper riêng và trọng số pha gắp. Đây là thay đổi
model/training trên dữ liệu đã có, không thu thêm demonstration. Chưa có kết quả
train đầy đủ hay bằng chứng success rate cao hơn checkpoint FlowPolicy.

## Kiến trúc và tương thích

- Hai depth view `chest_depth`, `front_depth`, cùng preprocessing v1: depth+mask,
  CNN dùng chung, giữ riêng đặc trưng từng camera; không cần calibration để train.
- 4 observation tại 50 Hz, horizon 16, trả 4 action. Input không đọc phase mask.
- MeanFlow sinh **14 khớp tay** theo residual so với qpos hiện tại; một lần gọi UNet
  khi infer. Nhánh MLP riêng sinh chuỗi độ mở hai gripper bằng sigmoid ×0.043 m.
  Gripper phụ thuộc observation, không phụ thuộc action noise.
- Widths `[64,128,256]`: **6,775,262 tham số** (bao gồm encoder và gripper head).
  So `[128,256,512]` bằng run mới nếu cần; tăng model chưa được chứng minh hữu ích.
- Inference trả `[4,16]` absolute targets, đúng thứ tự khớp hiện có. Gripper 0–43 mm,
  arm targets giới hạn theo model V1 với cấu hình v2 mặc định.
- Release ghi `policy_type=meanflow_depth_v2`; Python adapter
  `OpenArmMeanFlowPolicy` đọc cả v1/v2. Kiến trúc v2 không load weight FlowPolicy/v1.
  `conf/meanflow.yaml` giữ cấu hình v1; mặc định trainer mới dùng `conf/meanflow_v2.yaml`.

## Loss và pha gắp

```text
L = L_meanflow + 0.01 L_dispersive
  + 0.5 L_action + 1.0 L_gripper
  + 0.1 L_TCP_position + 0.02 L_TCP_orientation
```

Các hệ số trên là điểm khởi đầu để kiểm chứng, không phải hyperparameter đã tối ưu.

- MeanFlow Identity/JVP giữ nguyên trên 14 arm channels.
- Một forward bổ sung ở `t=1,r=0` huấn luyện **đúng đường sinh action lúc deploy**.
  Loss action dùng smooth L1 trên residual đã chuẩn hóa; không chỉ tối ưu velocity.
- FK PyTorch tính vị trí/hướng TCP hai tay từ endpoint dự đoán. Geometry export từ
  robot model vendored; không cần cài MuJoCo trên Colab. JSON geometry và license
  Apache-2.0 được kèm trong `src/openarm_flow/assets/`.
- Position loss smooth L1 theo sai số /0.03 m; orientation loss chordal theo ma trận
  rotation /0.25 rad, tránh gradient `acos` suy biến tại rotation trùng nhau.
- TCP target dùng demonstration command được giới hạn như controller. FK prediction
  lúc train giữ gradient; inference có giới hạn đầu ra. Joint command labels của
  dataset không bị ghi lại. Gripper loss giới hạn target về stroke vật lý 0–43 mm.
- Pha lấy riêng cho từng gripper từ chuyển tiếp command mở→đóng dưới 30 mm:
  **1 giây trước +0.5 giây sau** được tăng trọng số 3. Episode đã đóng ngay từ đầu
  không tự coi là sự kiện đóng mới. Mask không đi qua ranh giới episode.
- Mask `action_approach` chỉ 1 giây trước sự kiện; `action_phase` gồm trước/sau.
  Cả hai chỉ là supervision/metric, không được đưa vào encoder hoặc inference.
- Toàn bộ loss loại target padding; thống kê vẫn chỉ fit train episodes.

FK là proxy theo cùng URDF, không phải TCP đo độc lập ngoài thực tế. V2 không sửa
được calibration/cảnh sai chỉ bằng thêm loss.

## Dataset cũ dùng được

Zarr hai camera từ converter MeanFlow v1 vẫn dùng được; phase mask được suy từ
nhãn command hiện có lúc đọc, không cần đổi nội dung dataset. Split và nhãn giữ
theo metadata. Không dùng Zarr một-camera FlowPolicy làm đầu vào v2.

```bash
.venv/bin/python scripts/convert_meanflow_dataset.py \
  --raw '/media/space/New Volume/dataset/2026-10-13' \
  --output data/meanflow_openarm_20261013.zarr \
  --action-source commands --skip-invalid
```

Converter ghi rõ file hỏng và lý do, từ chối output có sẵn. Cần hai depth camera
đúng thứ tự; không có chế độ giả camera front từ chest. `debug_subset` chỉ để smoke.

## Train và chọn checkpoint

```bash
.venv/bin/python scripts/train_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --output checkpoints/meanflow_v2_small \
  --config conf/meanflow_v2.yaml --device cuda
```

Mặc định 20,000 steps, batch 16, checkpoint mỗi 500 steps, FP32. Giảm batch trước
khi train run mới nếu thiếu VRAM. Resume giữ model/dataset và training config,
chỉ tổng số steps được phép đổi:

```bash
.venv/bin/python scripts/train_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --output checkpoints/meanflow_v2_small \
  --config conf/meanflow_v2.yaml --device cuda --resume --steps 30000
```

`best_release` chọn bằng score thấp nhất:

```text
approach TCP RMSE / 30mm
+ 0.1 × approach orientation RMSE / 0.25rad
+ 0.25 × approach gripper MAE / 5mm
```

Nếu validation chưa có approach sample, metric `val_selection_uses_approach=false`
và dùng các chỉ số toàn bộ làm fallback; các chỉ số approach là null, không phải 0.
Log có số lượng approach samples, TCP RMSE/P95, orientation và gripper MAE. Validation
lấy mẫu đều trên mọi episode giữ lại; không phải toàn bộ frame trừ khi số mẫu đủ lớn.
Model selection chỉ dùng **4 action sẽ thực thi**; training học đủ horizon 16.

`loss_curve.png` vẽ loss; `action_metrics.png` vẽ TCP/gripper/score. Loss có các thành
phần và đơn vị chuẩn hóa khác v1, không so độ lớn loss v1/v2 trực tiếp.

```bash
.venv/bin/python scripts/eval_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --checkpoint checkpoints/meanflow_v2_small/best_release \
  --device cuda --samples-per-episode 128 --output outputs/meanflow_v2_eval.json
```

Để tuyên bố cải thiện cần train đầy đủ, cùng split, nhiều noise seeds và rollout.
Chọn checkpoint theo validation không thay thế test set độc lập.

## Colab

Build gói code hiện tại, không cần đẩy lên GitHub:

```bash
.venv/bin/python scripts/build_meanflow_colab.py
```

Output `outputs/meanflow_v2_colab/`:

- `OpenArm_MeanFlow_v2_Colab.ipynb`: mở bằng Upload notebook trên Colab.
- `notebooks/OpenArm_MeanFlow_v2_Colab.ipynb`: cùng notebook, nằm trực tiếp trong repo.
- `meanflow_v2_colab.tar.gz`: bản bundle dự phòng, không cần upload khi dùng notebook mới.
- `SOURCE_SHA256.json`, `SHA256SUMS.txt`: kiểm tra source/bundle tương ứng.

Notebook mount Drive rồi clone `https://github.com/vanphong864/openarm_act.git`.
Snapshot source hiện tại được nhúng trong notebook, dùng khi checkout khác bản đã
kiểm tra (các thay đổi local chưa được push cũng chạy được). Không cần tar riêng.
Đặt `USE_NOTEBOOK_SNAPSHOT=False` nếu chủ động dùng code GitHub mới hơn; restart
runtime khi đổi code. Notebook không push repo hay khởi tạo GPU bên ngoài Colab.

HDF5 mặc định: `/content/drive/MyDrive/vrbt/dataset/2026-10-13`.
Nếu chỉ thấy folder trong “Được chia sẻ với tôi”, thêm shortcut vào `MyDrive/vrbt`.
Convert hai camera `chest_depth`, `front_depth` với `--skip-invalid`, report file
hỏng vào metadata. Zarr trên `/content`, cache và checkpoint tại
`MyDrive/vrbt/meanflow_v2/2026-10-13/` (cache riêng cho dataset này).
Có GPU smoke 2 steps trước train 20,000 steps, log trực tiếp, tự resume,
best EMA release, evaluation và tải ZIP. Giữ torch CUDA của Colab.
Phiên local không có CUDA; GPU smoke vẫn cần chạy trên Colab thực tế.

## Kiểm chứng

```bash
.venv/bin/python -m pytest tests/test_meanflow.py tests/test_meanflow_v2.py \
  tests/test_flow.py tests/test_episode.py -q
```

Test mới kiểm tra FK so MuJoCo trên 10 pose ngẫu nhiên, gradient theo sai phân hữu
hạn, pha episode-local, gradient về encoder/arm/gripper, mọi auxiliary loss loại
padding, đúng 1 UNet inference, gripper không đổi khi chỉ thay noise, release v2,
train/resume/eval và checkpoint selection theo approach.
MuJoCo test chỉ chạy tại workspace có simulator; Colab bundle bỏ simulator và skip
test đó. Test còn lại không cần MuJoCo.

Các smoke/checkpoint trong `outputs/meanflow_v2_smoke/` chỉ xác minh kỹ thuật,
không dùng điều khiển robot hay đánh giá chất lượng gắp.

Kiểm chứng tại workspace ngày 2026-10-16: **29 tests passed**. Smoke dùng hai
episode thật `105708` (validation) và `105835` (train), train 3 steps rồi resume
đến step 4 trên CPU. Evaluation 64 chunks ghi nhận 20 approach samples; inference
đọc trực tiếp hai depth view ở 4 frame thật trả `[4,16]`, hữu hạn và đúng giới hạn
khớp. Các kết quả này kiểm tra pipeline, không chứng minh chất lượng sau train.
Máy này không có CUDA; notebook có GPU smoke riêng trước run chính.

## Nhánh RGB (tùy chọn)

Depth không phân biệt được vật mỏng trên bàn và tờ giấy trắng là lỗ depth, nên có thể
thêm hai camera RGB (`chest_rgb`, `front_rgb`).

- Convert: `scripts/convert_meanflow_dataset.py --rgb` lưu thêm `data/rgb` (uint8,
  `[N,2,3,H,W]`, cùng lưới với depth) và `rgb` trong metadata. Dung lượng ~130 KB/frame.
- Model: `model.use_rgb: true` thêm một CNN riêng cho RGB (late fusion, không giả định
  depth/RGB căn chỉnh pixel); augment brightness/contrast và view dropout. Dataset và
  `use_rgb` phải khớp, train script báo lỗi nếu lệch.
- Release/`OpenArmMeanFlowPolicy.observe` cần thêm `images[chest_rgb|front_rgb]`.
- Đây là run mới: checkpoint depth-only không load được. Chưa có bằng chứng thực nghiệm
  RGB cải thiện; cần so sánh validation (TCP/gripper) với run depth-only.

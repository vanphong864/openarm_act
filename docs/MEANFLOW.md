# MeanFlow cho OpenArm V1, hai camera depth cố định

**Bản v2 mới:** xem [MEANFLOW_V2.md](MEANFLOW_V2.md) để dùng TCP/action loss,
nhánh gripper riêng và notebook Colab mới. Tài liệu dưới mô tả v1; muốn chạy đúng
cấu hình v1, truyền `--config conf/meanflow.yaml` (trainer mặc định hiện là v2).

Implementation mới nằm trong `src/openarm_flow/meanflow*.py` và `depth_views.py`.
Giữ pipeline FlowPolicy hiện có để đối chứng. MeanFlow có model, dataset và release
riêng; checkpoint FlowPolicy không tương thích trực tiếp, cần train lại.

## Thiết kế theo dữ liệu hiện có

Hai HDF5 đọc được trong ba mẫu người dùng cung cấp có `chest_depth` và `front_depth`
uint16 480×640, scale 0.001 m/unit, giới hạn depth ghi nhận 0.2–1.2 m, qpos 16 chiều,
50 Hz. Người dùng xác nhận hai camera cố định. HDF5 không chứa calibration.

```text
chest depth ── masked resize ── shared depth CNN ── chest features ──┐
front depth ── masked resize ── shared depth CNN ── front features ──┤
qpos history ────────────────── state MLP ─────────────────────────┤
                                                                ↓
Gaussian noise [16,16] + times (t,r) ── conditioned MeanFlow U-Net
                                                                ↓
                     noise − velocity → residual arm / absolute grippers
                                                                ↓
                        absolute targets [8,16] for the controller
```

- Depth giảm xuống 120×160 bằng trung bình vùng **chỉ trên pixel hợp lệ**. Pixel
  bằng 0 hoặc chạm biên clip bị loại; vùng thiếu coverage giữ invalid. Encoder đọc
  depth chuẩn hóa, mask và tọa độ pixel chuẩn hóa. Không chuyển thành XYZ giả.
- CNN dùng chung trọng số cho hai camera, GroupNorm (hợp batch nhỏ), pooling giữ
  lưới không gian 3×4. Ghép feature theo thứ tự camera, không lấy trung bình hai view.
- Lịch sử 2 observation ở 50 Hz; state MLP và thứ tự lịch sử cung cấp thông tin
  chuyển động. Input không sử dụng RGB, effort hay qvel.
- U-Net widths `[64,128,256]`, **4,805,568 tham số** với cấu hình mặc định. Nhỏ hơn
  bản FlowPolicy đang dùng, là điểm bắt đầu cho khoảng 100 demonstration, chưa tuning.
- Sinh horizon 16 bước (0.32 s), trả 8 bước (0.16 s). Chunk bắt đầu tại observation
  hiện tại; không có prefix action trong quá khứ. Khi chạy robot có thể replan sớm hơn
  bằng cách thực thi ít hàng hơn, vẫn cập nhật observation mỗi tick.
- Arm targets học dưới dạng `target − qpos_current`; hai gripper học giá trị tuyệt đối
  bằng mét. API trả toàn bộ 16 target tuyệt đối đúng thứ tự khớp repo.
- Noise depth ~3 mm, pixel dropout 2%, có xác suất 10% bỏ một view trong cả history
  khi train; inference yêu cầu đủ cả hai camera. Không flip ảnh hoặc đổi thứ tự camera.

Calibrations không bắt buộc cho kiến trúc này. Camera phải giữ nguyên vị trí, thứ tự,
grid depth/alignment và scale lúc train và inference. Nếu thay vị trí camera, cần dữ
liệu tương ứng hoặc một thiết kế dùng calibration; chỉ đổi intrinsics không sửa được
phân phối quan sát của model này.

## MeanFlow và mối liên hệ với MP1

Nguồn phương pháp:

- [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
  và [implementation của tác giả](https://github.com/Gsunshine/meanflow).
- [MP1](https://github.com/LogSSim/MP1), bài AAAI-26 người dùng cung cấp.
- Các residual block, convolution và time embedding được tái sử dụng từ FlowPolicy
  đã vendor trong repo; giấy phép và nguồn tại `src/openarm_flow/_vendor/`.

Đây là bản thích nghi cho OpenArm, không phải bản sao đầy đủ thí nghiệm MP1:

1. `z_t = (1-t) * action_normalized + t * noise`, `v = noise − action_normalized`.
2. Sample `0 ≤ r ≤ t ≤ 1`, với 50% xác suất `r=t` để học cả instantaneous flow.
3. JVP theo hướng `(v,1,0)` tính `d u / dt` với condition cố định.
4. Target `stopgrad(v − (t-r) * du/dt)`; forward riêng tính prediction để gradient
   truyền về cả UNet lẫn depth encoder. Không backprop qua target.
5. Adaptive L2 chỉ tính các timestep thật; padding cuối episode không tính loss.
6. Dispersive loss nhẹ trên feature các down block: normalize feature, loại self-pair,
   log-mean-exp khoảng cách âm. Có bảo vệ batch=1 và feature trùng nhau, trọng số 0.01.
7. Inference gọi UNet **đúng một lần**: `a = noise − u(noise,t=1,r=0,condition)`.

Không bật CFG ở bản đầu; condition trực tiếp là hai view và qpos. Dispersive loss là
biến thể ổn định số, không trùng mọi chi tiết triển khai MP1. Có thể đặt
`dispersive_weight: 0` để ablation. Đây là các lựa chọn cần đánh giá trên dataset này,
không bảo đảm tự động tăng success rate so với FlowPolicy.

## Dữ liệu và chuyển đổi

Trong hai mẫu đọc được, `action` bằng qpos cùng thời điểm. Mặc định dùng
`joint_commands` qua logic `EpisodeReader.labels("commands")`: cắt phần đầu thiếu
command, nội suy thiếu nhỏ, fallback sang next_qpos nếu thiếu nhiều; cột không được
command dùng next_qpos. Source thực tế và notes được lưu theo episode. Không xem
qpos cùng thời điểm là nhãn chuyển động trong cấu hình mặc định.

Một lần audit tại máy này thấy 100 HDF5, **88 header đọc được, 12 file không mở được**.
Đó chỉ là audit header/shape, chưa chứng minh mọi frame payload của 88 file hợp lệ.
Mẫu `episode_20261013_105807.hdf5` bị truncated: kích thước file khoảng 511 MB, header
khai báo khoảng 1,126 MB. Cần chép lại file đầy đủ để dùng episode đó.
Kết quả máy này: `outputs/meanflow_smoke/raw_header_audit.json`.

Cài dependency trong môi trường repo (không cần MetaWorld hoặc MuJoCo để train):

```bash
uv pip install --python .venv/bin/python -r requirements-flow.txt
```

Convert toàn bộ dữ liệu:

```bash
.venv/bin/python scripts/convert_meanflow_dataset.py \
  --raw '/media/space/New Volume/dataset/2026-10-13' \
  --output data/meanflow_openarm_20261013.zarr \
  --action-source commands --skip-invalid
```

`--skip-invalid` ghi rõ từng episode bị loại và lý do trong Zarr metadata. Nếu bỏ cờ,
converter dừng khi có lỗi. File nguồn chỉ đọc. Output có sẵn bị từ chối; conversion
bị ngắt có `complete=false`, không được trainer sử dụng. Converter kiểm tra từng
frame được chuyển đổi; nếu cả một split bị loại, dataset không đánh dấu complete.

- Split theo **episode**, mặc định khoảng 85% train / 15% validation, seed 42. Có thể
  chỉ định chính xác bằng `--manifest` với `episodes.<id>.split: val`.
- Episode ID là đường dẫn tương đối với `--raw`, bỏ `.hdf5`.
- Mask, chunk và observation history không đi qua ranh giới episode.
- Mean/std state và thống kê residual action tính từ train, không dùng validation
  hoặc target padding. Có sàn scale cho khớp ít chuyển động.
- Zarr chứa `data/{depth,state,action,source_index}`, `meta/episode_ends`, camera order,
  source resolutions, preprocessing, labels và split. Depth lưu uint16 đã resize.
- Khoảng 76.8 KB depth thô/frame trước nén; 100×1000 frame khoảng 7.7 GB trước nén.
- `--max-frames`, `--max-episodes`, `--episodes` dùng cho smoke, đánh dấu `debug_subset`.

## Train, resume và đánh giá

```bash
.venv/bin/python scripts/train_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --output checkpoints/meanflow_openarm_v1 --device cuda
```

Mặc định 20,000 steps, batch 16, AdamW lr 1e-4, weight decay 1e-4, warmup 500,
gradient clipping 1.0, EMA. FP32 cho JVP; chưa tích hợp AMP. Giảm `--batch-size 8`
nếu thiếu VRAM. Config đầy đủ: `conf/meanflow.yaml`, chọn khác bằng `--config`.

```bash
# Resume cùng cấu hình, optimizer, model, EMA và RNG.
.venv/bin/python scripts/train_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --output checkpoints/meanflow_openarm_v1 --device cuda --resume --steps 30000

.venv/bin/python scripts/eval_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --checkpoint checkpoints/meanflow_openarm_v1/best_release \
  --device cuda --samples-per-episode 128 \
  --output outputs/meanflow_eval.json
```

Resume kiểm tra metadata và cấu hình; chỉ tổng số steps được phép đổi. Đổi steps làm
đổi lịch cosine. Thứ tự minibatch không khôi phục đúng giữa epoch; chưa phải bitwise
resume. `best_release` dùng validation normalized action MSE thấp nhất, không chọn
theo train loss. Validation lấy mẫu đều trên **mọi** held-out episode, có mask padding,
noise seed cố định, báo arm RMSE (rad), gripper MAE (mm) và baseline giữ nguyên qpos.
Không dùng validation như test set độc lập để tuyên bố kết quả cuối cùng.

Run xuất `last.pt`, `release/` (EMA cuối), `best_release/`, `best_metrics.json`,
`train_config.yaml`, `train_loss.jsonl`, `metrics.jsonl` và `loss_curve.png` nếu có
matplotlib. Train và val loss có noise/augmentation khác; so chất lượng bằng action
metrics. Cần đánh giá nhiều seed và closed-loop task success trên robot trước khi
kết luận tốt hơn. So sánh các model bằng cùng split và cùng nhãn action.

## Python inference API

```python
from openarm_flow.meanflow_policy import OpenArmMeanFlowPolicy

policy = OpenArmMeanFlowPolicy("checkpoints/meanflow_openarm_v1/best_release", device="cuda")
policy.reset()  # mỗi episode mới

# Mỗi tick 50 Hz, kể cả khi vẫn đang thực thi chunk:
policy.observe({
    "images": {"chest_depth": chest_uint16, "front_depth": front_uint16},
    "state": qpos_float32_16,
})
actions = policy.infer()["actions"]  # [8,16], absolute joints rad / grippers m
```

Scale và resolution phải khớp metadata; cả hai frame phải được đồng bộ với qpos theo
cách recorder đã dùng. `reset()` pad history bằng observation đầu và reset RNG riêng.
Mỗi query lấy noise mới; cùng observation và seed sau reset cho cùng kết quả.
Controller vẫn chịu trách nhiệm giới hạn khớp, tốc độ và thực thi chunk.
Runner MuJoCo một camera hiện có chưa hỗ trợ release này;
API trên là điểm tích hợp hai camera.

## Kiểm chứng tại máy này

```bash
.venv/bin/python -m pytest tests/test_meanflow.py tests/test_flow.py tests/test_episode.py -q
```

Các test kiểm tra JVP với nghiệm giải tích, stop-gradient, gradient về hai view/time r,
đúng 1 lần gọi UNet khi infer, invalid depth, frame/camera order, episode boundaries,
padding mask, train-only stats, file hỏng, roundtrip release, preprocessing train/serve,
train/resume/eval CLI và regression FlowPolicy cũ.

Smoke thực dùng hai episode `105708`, `105835` (64 labelled frames mỗi episode),
ghi nhận và bỏ qua mẫu `105807` hỏng; chạy model kích thước mặc định trên CPU.
Kết quả: **22 tests passed**; train 3 steps, resume đến step 4, eval từ release và
inference với hai frame thật trả `[8,16]` hữu hạn. Smoke arm RMSE 0.0909 rad, baseline
hold 0.0663 rad: checkpoint vài bước này chưa tốt hơn baseline.
Artifact tại `outputs/meanflow_smoke/`. Model smoke chỉ kiểm tra kỹ thuật, không dùng
điều khiển robot hay đánh giá khả năng thao tác. Chưa train đầy đủ vì môi trường này
không có CUDA khả dụng.

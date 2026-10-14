# FlowPolicy cho OpenArm

Pipeline mới dùng depth → XYZ (m) trong hệ camera quang học → PointNet,
ghép với MLP qpos → conditional U-Net 1D → consistency flow matching.
State và action có 16 chiều: trái 7 khớp + gripper, phải 7 khớp + gripper;
khớp tính bằng radian, gripper bằng mét. Action là joint target tuyệt đối.

Code model/loss lấy từ FlowPolicy commit `86d10c14972b4eab083abec9026749c340f8e5ed`,
có giấy phép MIT và ghi rõ thay đổi trong `src/openarm_flow/_vendor/SOURCE.md`.
Không cần cài bộ MetaWorld, mujoco-py hay Python 3.8 của repo nguồn.

## Cấu hình đang dùng

- Chỉ XYZ mặc định, 512 điểm, lọc depth ngoài (0.2, 1.2) m và pixel không hợp lệ.
- Lấy mẫu đều không hoàn lại; lặp điểm nếu còn ít hơn 512 điểm; lỗi nếu dưới 32 điểm.
- 2 observation liên tiếp tại 50 Hz; horizon 16, thực thi 8 action trước khi dự đoán lại.
- PointNet feature 64 + qpos feature 64; U-Net widths `[128,256,512]`, khoảng 18.8M tham số.
- Đây là bản cấu hình nhỏ cho OpenArm. Widths mặc định của repo gốc là `[512,1024,2048]`;
  có thể chỉnh trong `conf/flowpolicy.yaml` khi cần so sánh.
- Infer một bước flow từ Gaussian noise; train giữ nguyên consistency loss của nguồn.
- EMA để infer; validation báo lỗi khớp/gripper và baseline giữ nguyên qpos.
- Chuỗi luôn nằm trong cùng episode, padding lặp frame ở đầu/cuối.
  Horizon phải chia hết cho hệ số downsample U-Net và ≥ obs_steps + action_steps - 1.

## Camera chưa có intrinsics

HDF5 đang có depth 640×480, scale 0.001, nhưng không lưu fx/fy/cx/cy.
`conf/flow_camera.approximate.yaml` **chỉ là ước lượng** từ camera MuJoCo:
FOV dọc 42°, render 1280×720, resize về 640×480. Nó không phải calibration đo
trên camera thật. File được đánh dấu APPROXIMATE, lưu kèm dataset và release.
Converter yêu cầu `--allow-approximate-camera` để dùng rõ ràng cho thí nghiệm.
Các kiểm thử đã qua chứng minh code hoạt động, không chứng minh XYZ đúng với cảnh thật.

Nếu lấy được CameraInfo/calibration, điền `conf/flow_camera.example.yaml` và dùng file
đó, bỏ cờ approximate. Cần intrinsics của chính grid depth đã ghi: nếu depth align
sang RGB thì dùng intrinsics RGB tương ứng, có điều chỉnh resize/crop. Dữ liệu đầu
vào phải rectified; adapter chưa hỗ trợ distortion chưa sửa. Sau khi thay calibration,
convert ra dataset mới và train lại. Tọa độ dùng x sang phải, y xuống, z về trước;
không cần extrinsics để giữ mọi dữ liệu trong cùng camera cố định, nhưng việc di chuyển
camera giữa thu data và infer sẽ làm thay đổi phân phối.

## Cài đặt

Từ thư mục gốc repo, trong môi trường đã dùng train ACT:

```bash
uv pip install --python .venv/bin/python -r requirements-flow.txt
```

Các script FlowPolicy dùng trực tiếp `.venv/bin/python`. Nếu chạy `uv sync` về sau,
cài lại `requirements-flow.txt` vì các dependency tùy chọn này không nằm trong lock ACT.
MuJoCo dùng thêm `simulation/requirements.txt` nếu chưa cài.

## Convert toàn bộ dữ liệu đã tải

```bash
.venv/bin/python scripts/convert_flow_dataset.py \
  --raw data/raw/data_openarm_8_10 \
  --output data/flow_openarm_v1.zarr \
  --camera conf/flow_camera.approximate.yaml \
  --allow-approximate-camera \
  --action-source commands
```

Chỉ đọc HDF5 nguồn. Output tồn tại sẽ bị từ chối để tránh trộn các dataset.
Nếu convert bị ngắt, dataset có `complete=false` và trainer từ chối nó; chọn output mới
để chạy lại. Không có chế độ resume conversion.

Mặc định đọc mọi episode hoàn chỉnh, giữ exclusions/validation từ `manifests/episodes.yaml`.
Nếu manifest chưa có validation, chọn khoảng 15% episode bằng hash ổn định; không chia
ngẫu nhiên theo frame. Thống kê chuẩn hóa chỉ fit trên train, tính theo từng block để
không phải đưa toàn bộ point cloud vào RAM.

`commands` dùng logic labels sẵn của repo: joint_commands khi có, các cột không được
command (ví dụ gripper) dùng next_qpos; nếu thiếu nhiều thì cả episode dùng next_qpos.
Dataset ghi source thực tế và ghi chú cho từng episode. `--action-source qpos` có sẵn
để đối chứng với bản ACT cũ, nhưng đó là trạng thái đo ở cùng thời điểm.

Tùy chọn:

- `--use-color`: gắn màu RGB/255 cho từng điểm XYZ; phải có RGB cùng grid với depth.
- `--xyz-min X Y Z --xyz-max X Y Z`: crop workspace trong hệ camera, đơn vị mét.
- `--max-episodes 3 --max-frames 64`: kiểm tra nhanh, không dùng cho full train.
- `--fps 25`: giảm tần số; runner MuJoCo hiện yêu cầu cùng 50 Hz nên mặc định giữ 50.

Zarr có `data/{point_cloud,state,action,source_index}`, `meta/episode_ends` và metadata
calibration, splits, labels, joint order. Chỉ RGB khi có `--use-color`; không cần RGB
cho dataset depth-only. Depth clip ở hai biên bị loại vì không còn đáng tin về khoảng cách.

## Train trên GPU

```bash
.venv/bin/python scripts/train_flowpolicy.py \
  --dataset data/flow_openarm_v1.zarr \
  --output checkpoints/flow_openarm_v1 \
  --device cuda --steps 40000 --batch-size 32
```

Nếu thiếu VRAM, giảm batch size. Thông số architecture, learning rate và interval nằm
trong `conf/flowpolicy.yaml`. Đây là điểm bắt đầu thử nghiệm, chưa là cấu hình đã tối ưu
cho dataset của bạn. Không tự động khởi động GPU hay train dài khi tích hợp code.

Chạy lại cùng lệnh và thêm `--resume` để tiếp tục từ `last.pt`. Có optimizer, model,
EMA và torch RNG; thứ tự minibatch có thể khác sau resume. Đổi `--steps` làm thay đổi
lịch cosine còn lại. Không resume nếu metadata dataset/model khác.

Mỗi 2000 bước và cuối run ghi validation, `last.pt` và cập nhật release EMA mới nhất:

```text
checkpoints/flow_openarm_v1/
  last.pt                 # model + EMA + optimizer để resume
  metrics.jsonl           # validation và baseline hold
  train_config.yaml
  release/
    model.pt              # EMA weights + normalizer
    config.json
    metadata.json         # calibration/point sampling/fps/joints/splits
```

Release là checkpoint cuối, chưa tự chọn best hoặc lưu mọi mốc.
Tải toàn bộ `release/` về máy để infer. Weight ACT đã train không chuyển trực tiếp sang FlowPolicy.

## Infer trong MuJoCo

```bash
.venv/bin/python scripts/rollout_mujoco.py \
  --policy flowpolicy \
  --checkpoint checkpoints/flow_openarm_v1/release \
  --seconds 10 --out outputs/flow_openarm_v1
```

Thêm `--view` để mở cửa sổ. Có GIF, ảnh RGB cuối, trajectory và result.json.
Runner cập nhật depth + qpos mỗi tick 50 Hz để giữ đúng 2-frame history, kể cả khi
đang thực thi chunk. CPU có thể chạy chậm hơn thời gian thực. Joint targets được
giới hạn bởi servo MuJoCo; trajectory lưu action dự đoán trước khi clip.

Runner hiện hỗ trợ camera giữa ảnh với aspect ratio tương ứng native 1280×720 →
640×480. Nếu calibration khác, runner báo lỗi thay vì dùng sai phép chiếu; cần sửa
camera/render cho đúng calibration đó. FOV dọc được lấy từ fy của release.

Python API cho tích hợp khác:

```python
from openarm_flow.policy import OpenArmFlowPolicy
policy = OpenArmFlowPolicy("checkpoints/flow_openarm_v1/release")
policy.reset()  # đầu episode
# Gọi mỗi tick tại fps của dataset:
policy.observe({"images": {"chest_depth": depth_uint16_mm}, "state": qpos16})
# Khi cần chunk mới:
actions = policy.infer()["actions"]  # [8,16] targets tuyệt đối
```

Với XYZRGB, thêm `chest: rgb_uint8` vào images. API này và runner đã hỗ trợ
FlowPolicy; websocket/ROS client ACT cũ chưa được đổi sang giao thức history ảnh
của FlowPolicy, không dùng `scripts/serve.py` cho release này.

## Kiểm chứng đã thực hiện

Unit tests kiểm tra pinhole + invalid mask + màu, padding và boundaries episode,
normalizer không dùng validation, loss/backprop cập nhật weight và roundtrip release.
Đã convert 3 episode thực (64 frame/episode với intrinsics ước lượng), train 3 bước,
resume đến bước 4 và chạy 20 bước vật lý MuJoCo với checkpoint smoke. Đây là kiểm
tra end-to-end; checkpoint smoke không dùng đánh giá chất lượng gắp.

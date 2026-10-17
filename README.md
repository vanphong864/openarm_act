# OpenArm MeanFlow / FlowPolicy

Sinh joint action cho OpenArm V1 từ dữ liệu teleoperation HDF5. Pipeline chính là
**MeanFlow v2**, dùng hai camera depth cố định `chest_depth` và `front_depth`.
FlowPolicy một camera được giữ làm baseline. Repo đã bỏ model, pipeline train,
export, server và client ACT; phần đọc dữ liệu dùng chung nằm tại `openarm_data`.

State/action có 16 chiều: trái 7 khớp + gripper, phải 7 khớp + gripper.
Khớp dùng radian, gripper dùng mét. Policy trả absolute joint targets.

## MeanFlow v2

- Hai depth view có mask, lịch sử 4 observation tại 50 Hz.
- MeanFlow sinh 14 khớp tay, nhánh riêng sinh hai gripper.
- Loss action và FK TCP position/orientation, tăng trọng số pha gắp.
- Cấu hình nhỏ khoảng 6.8 triệu tham số, horizon 16, thực thi 4 action mỗi chunk.
- Chọn checkpoint theo TCP/gripper trong pha tiếp cận.

Xem [hướng dẫn MeanFlow v2](docs/MEANFLOW_V2.md) và
[cấu hình train](conf/meanflow_v2.yaml). FK là proxy theo model robot;
chưa có kết quả train đầy đủ chứng minh tỷ lệ gắp cao hơn FlowPolicy.

## Train trên Colab

Build gói từ code hiện tại:

```bash
.venv/bin/python scripts/build_meanflow_colab.py
```

1. Mở `notebooks/OpenArm_MeanFlow_v2_Colab.ipynb` trên Colab (hoặc upload bản trong outputs).
2. Chọn GPU rồi **Run all**. Notebook clone `https://github.com/vanphong864/openarm_act.git`.
3. Dữ liệu mặc định: `/content/drive/MyDrive/vrbt/dataset/2026-10-13`.
   Nếu folder ở “Được chia sẻ với tôi”, thêm shortcut vào `MyDrive/vrbt`.

Notebook kèm snapshot code hiện tại để clone vẫn chạy được khi GitHub chưa có v2;
không cần upload tar riêng. Có GPU smoke, convert/cache Zarr, tự resume, evaluation
và tải best release. Checkpoint lưu tại
`MyDrive/vrbt/meanflow_v2/2026-10-13/checkpoints/meanflow_v2_small`.
Dependency cài từ `requirements-colab.txt`, giữ torch CUDA của Colab.
Checkpoint FlowPolicy/MeanFlow v1 không load trực tiếp vào kiến trúc v2.

## Cài và train tại máy

Python 3.12. Cài dependency vào môi trường có torch phù hợp với CPU/CUDA của máy:

```bash
python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e '.[calibration]' pytest
```

Extra `calibration` chỉ cần cho các công cụ căn camera. Có thể bỏ extra khi chỉ train.
Dependency model/dataset khai báo trong `pyproject.toml`; không cần LeRobot.
Các lệnh CLI thêm `src/` vào Python path khi chạy trực tiếp.

```bash
.venv/bin/python scripts/convert_meanflow_dataset.py \
  --raw '/media/space/New Volume/dataset/2026-10-13' \
  --output data/meanflow_openarm_20261013.zarr \
  --action-source commands --skip-invalid

.venv/bin/python scripts/train_meanflow.py \
  --dataset data/meanflow_openarm_20261013.zarr \
  --output checkpoints/meanflow_v2_small \
  --config conf/meanflow_v2.yaml --device cuda
```

Thêm `--resume` để tiếp tục từ `last.pt`. Best EMA release gồm `model.pt`,
`config.json`, `metadata.json`. Converter chỉ đọc HDF5, ghi rõ file hỏng;
không dùng subset smoke cho train chính. Split theo episode, thống kê chỉ fit train.

## FlowPolicy và mô phỏng

- [FlowPolicy: convert, train và infer](docs/FLOWPOLICY.md).
- [MuJoCo workspace và rollout FlowPolicy](simulation/README.md).
- [Căn camera/cảnh bằng số đo độc lập](docs/SCENE_CALIBRATION.md).
- [MeanFlow v1](docs/MEANFLOW.md) giữ cho thí nghiệm cũ.

Runner MuJoCo hiện hỗ trợ release FlowPolicy một camera; release MeanFlow hai
camera cần adapter tương ứng. Không dùng ảnh chest giả làm ảnh front.

## Cấu trúc và kiểm thử

```text
src/openarm_data/        HDF5 reader, nhãn action, joint constants, episode manifest
src/openarm_flow/        FlowPolicy, MeanFlow, encoders, FK và adapters
conf/                   cấu hình MeanFlow/FlowPolicy và camera
scripts/                convert, train, evaluate, calibration, rollout, build Colab
simulation/             robot model, scene và MuJoCo workspace
manifests/              task/split/exclusions theo episode
```

```bash
.venv/bin/python -m pytest -q
```

Test FK so MuJoCo cần `simulation/requirements.txt`; test calibration cần OpenCV.
Dữ liệu, checkpoint, release và output hiện có không bị xóa khi dọn pipeline ACT.
Giấy phép code FlowPolicy và robot geometry được giữ tại `_vendor/` và `assets/`.

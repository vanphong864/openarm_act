# openarm_act — Thiết kế

Train **ACT** (Action Chunking with Transformers) trên dữ liệu teleop của
OpenArm bimanual, với backbone thị giác chọn được — **DINOv3**, **DINOv2**
hoặc **ResNet** của chính ACT — rồi chạy trên robot qua đúng stack mà
SmolVLA ([openarm_smolvla](../../openarm_smolvla/docs/DESIGN.md)), π0/π0.5
và LingBot đang dùng.

> Trạng thái (12/10/2026): code đầy đủ từ dữ liệu tới robot, cùng ba cách
> xử lý depth mới (mục 3.4) và các cờ train (`--resnet`, `--dinov2`,
> `--lingbot-depth`, ...). LeRobot pin ở **0.6.1**, các repo pin theo commit.
> Đã chạy thông trên CPU: 108 test, và pipeline trên 2 episode thật 640×480
> với trọng số thật của DINOv2-S và LingBot-Depth (`--lingbot-depth`,
> `--lingbot-depth-distill`). Chưa train thật trên GPU, chưa thử trọng số
> DINOv3 thật (repo gated, máy dev chưa có quyền).

---

## 1. Bài toán và hợp đồng

Giống hệt openarm_smolvla: 16 DoF, một camera ngực RGB-D 640×480, ghi 50 Hz,
HDF5 của `openarm_ui/data_recorder.py`. Phần đọc episode, manifest, nhãn,
chia held-out, norm stats, delta actions được chép nguyên từ openarm_smolvla,
nên mọi model dùng **cùng dataset, cùng nhãn, cùng tập val**.

Phía robot giữ hợp đồng của `openarm_act`:

```python
policy.predict(rgb: uint8[H,W,3], depth: uint16[H,W], qpos: float32[16])
    -> float32[50, 16]      # target khớp tuyệt đối
```

## 2. Câu hỏi của repo này

act_pipeline (OpenArm_MC) là ACT đang chạy trên robot: ResNet-18 ImageNet,
`conv1` nới thành 4 kênh (RGB-D) và được train, phần còn lại đóng băng; 300
token ảnh (15×20) từ ảnh 640×480. Repo này giữ nguyên công thức đó và chỉ
thay backbone, để trả lời: **backbone tự giám sát mạnh hơn (DINOv3, DINOv2)
có làm ACT tốt hơn không?** `backbone=resnet18` tái tạo backbone của
act_pipeline làm baseline, trong cùng code, cùng dữ liệu, cùng cách đánh giá.

| | ACT (repo này) | SmolVLA | π0 / π0.5 |
|---|---|---|---|
| Kích thước | 52M (ResNet-18) – 344M (ViT-L); 126M mặc định | ~450M | ~3.3B |
| Pretrain | chỉ backbone ảnh; phần ACT train từ đầu | VLM + action expert | VLM + action expert |
| Prompt | không | có | có |
| Sinh action | một lần forward, z = 0, tất định | 10 bước flow matching | 10 bước flow matching |
| Loss | L1 + KL (CVAE) | flow matching | flow matching |

## 3. Kiến trúc

![Kiến trúc ACT + DINOv3 với input RGB-D](images/architecture.png)

Hình vẽ cấu hình mặc định (`backbone=dinov3_vitb16 finetune=frozen
data=openarm_rgbd`, khung 640×480); nguồn: [images/architecture.svg](images/architecture.svg).
Viền mảnh là dữ liệu, viền đậm là phần được train, viền nét đứt là phần đóng
băng, nền xám là phần chỉ có lúc train.

### Input và output

| | Tên | Kiểu, shape | Đơn vị, ý nghĩa |
|---|---|---|---|
| **Input** | `rgb` | uint8 [480, 640, 3] | ảnh màu camera ngực |
| | `depth` | uint16 [480, 640] | mm, 0 = không đo được |
| | `qpos` | float32 [16] | `left_j1..j7, left_gripper, right_j1..j7, right_gripper`; rad, gripper là hành trình ngón (m) |
| **Chỉ lúc train** | `action` | float32 [50, 16] | nhãn: `qpos[t : t+50]`, cùng thứ tự và đơn vị |
| | `action_is_pad` | bool [50] | các bước vượt quá cuối episode, bị loại khỏi loss |
| **Output** | `actions` | float32 [50, 16] | target khớp tuyệt đối cho 50 bước (1 s ở 50 Hz); hàng 0 là bước lúc chụp quan sát |

### Từng bước, với shape

| Bước | Shape | Train? |
|---|---|---|
| RGB ÷255, chuẩn hoá ImageNet | [3, 480, 640] | — |
| depth → m, kẹp 0.2–1.2 m, → 0..1 | [1, 480, 640] | — |
| ghép thành RGB-D, resize bilinear | [4, 480, 640] → [4, 240, 320] | — |
| patch embedding Conv2d(4→768, 16×16) | [300, 768] (+ CLS, 4 register) | có (lr 1e-5) |
| DINOv3 ViT-B/16, 12 block, RoPE, LayerNorm cuối | [305, 768] | đóng băng |
| bỏ CLS và register → feature map | [768, 15, 20] | — |
| conv 1×1 768→512 + position 2D sin | 300 token × 512 | có |
| qpos chuẩn hoá → Linear 16→512 | state token [1, 512] | có |
| z (train: từ CVAE; suy luận: 0) → Linear 32→512 | latent token [1, 512] | có |
| encoder Transformer 4 lớp, 8 head, FFN 3200 | [302, 512] | có |
| 50 query (0 + position học được) → decoder 1 lớp | [50, 512] | có |
| Linear 512→16 | [50, 16] (đã chuẩn hoá) | có |
| bỏ chuẩn hoá (mean/std của action), + state nếu delta | [50, 16] tuyệt đối | — |

Lúc train, CVAE encoder (4 lớp) đọc [CLS, qpos, a₀ … a₄₉] = 52 token, ra μ và
log σ² (32 chiều), lấy mẫu z = μ + σ·ε. Loss = L1(dự đoán, nhãn, bỏ bước pad)
\+ 10 · KL(N(μ, σ²) ‖ N(0, I)). Lúc suy luận không có CVAE, z = 0, nên cùng
một quan sát luôn cho cùng một chunk.

Số tham số: DINOv3 ViT-B/16 85.9M (đóng băng, trừ patch embedding 0.8M), phần
ACT 40.5M train từ đầu, tổng 126.4M. Backbone khác chỉ đổi cột trái của hình:

| Backbone | Ảnh vào model | Token ảnh | Lớp đầu 4 kênh |
|---|---|---|---|
| DINOv3 ViT/16 | 320×240 | 15×20 patch 16 px, 384–1024 chiều | patch embedding |
| DINOv2 ViT/14 | 280×210 | 15×20 patch 14 px, 384–1024 chiều | patch embedding |
| ResNet-18/34/50 | 640×480 (không resize) | map layer4 15×20, 512 / 2048 chiều | `conv1` (4→64) |

Với `data=openarm_rgb_depth`, depth là ảnh thứ hai đi qua cùng backbone, thêm
300 token (encoder thành 602 token); với `data=openarm_rgb` chỉ có RGB.

Phần ACT là `ACTPolicy` của LeRobot, không sửa. `OpenArmACTPolicy`
(`modeling_openarm_act.py`, đăng ký policy type `openarm_act`) chỉ đổi ba chỗ:

1. **Backbone** (`backbones.py`): trả về `{"feature_map": [B, C, h, w]}` như
   `IntermediateLayerGetter` của ResNet, nên phần chiếu 1×1 conv và position
   embedding 2D sin của ACT dùng y nguyên. Với ViT, feature map là patch token
   sau layer norm cuối (bỏ CLS và register token). Tham số nằm ở
   `model.backbone.*`, nên hai nhóm optimizer của ACT tự tách backbone ra với
   `optimizer_lr_backbone`.
2. **Lớp đầu 4 kênh**: `conv1` của ResNet hoặc patch embedding của ViT nới từ
   3 lên 4 kênh. Kênh depth khởi tạo `zero` (mặc định: đặc trưng ban đầu y như
   pretrained) hoặc `mean` (như `conv1_adapter` của act_pipeline).
3. **`prepare_images`**: resize ảnh về `image_size` (bilinear, half-pixel như
   act_pipeline/cv2), chuẩn hoá ImageNet cho RGB, depth giữ 0..1 (0.2–1.2 m)
   như act_pipeline, ghép depth làm kênh 4. Vì vậy normalizer của LeRobot để
   ảnh IDENTITY và `use_imagenet_stats=false`.

### 3.1 Số token và kích thước ảnh

ACT đặt token ảnh, state và latent vào cùng một encoder, nên số token quyết
định chi phí và cân bằng giữa ảnh và state. Mọi backbone được đưa về **~300
token** như act_pipeline (`model.image_tokens`), giữ tỉ lệ khung:

| Backbone | Stride | 640×480 → | Token |
|---|---|---|---|
| DINOv3 | 16 | 320×240 | 20×15 = 300 |
| DINOv2 | 14 | 280×210 | 20×15 = 300 |
| ResNet | 32 | 640×480 (không resize) | 20×15 = 300 |

Khung 424×240 (recorder mới) → DINOv3 368×208 (23×13). `model.image_size`
đặt tay được. Kích thước được tính lúc train từ dataset và lưu trong
`config.json`, nên server không cần dataset.

### 3.2 Backbone

| Họ | Nguồn | Ghi chú |
|---|---|---|
| DINOv3 (`DINOv3ViTModel`) | `facebook/dinov3-*-pretrain-lvd1689m`, **gated, Meta duyệt tay** | RoPE, 4 register token. Tắt augmentation toạ độ RoPE (`pos_embed_rescale` …) của pretrain, để train và suy luận cùng hình học |
| DINOv2 (`Dinov2Model`) | `facebook/dinov2-{small,base,large}`, không gated | position embedding nội suy bicubic theo kích thước ảnh |
| ResNet | torchvision ImageNet, chính ACT của LeRobot dựng | FrozenBatchNorm; được bọc để theo cùng quy tắc trainable |

Config kiến trúc của ViT (`backbone_config`) được đọc từ `config.json` của
repo lúc bắt đầu train và lưu trong checkpoint; trọng số backbone nằm trong
`model.safetensors`. Khi nạp checkpoint (`from_pretrained`) không tải gì từ
Hub hay torchvision, nên máy serve **không cần quyền repo gated**. Nạp
checkpoint là strict: thiếu hay thừa tensor nào là báo lỗi, không bao giờ để
một lớp ở trọng số ngẫu nhiên điều khiển tay robot.

### 3.3 Các chế độ fine-tune

| `finetune` | Backbone | Ghi chú |
|---|---|---|
| `frozen` (mặc định) | đóng băng, trừ lớp đầu 4 kênh | công thức act_pipeline; ViT/ResNet không có gì để train thì chạy dưới `no_grad` |
| `partial` | + 4 block ViT cuối và norm cuối (ResNet: N stage cuối) | lr 1e-5 |
| `full` | mọi trọng số, lr 1e-5 | như ACT gốc train ResNet; dễ overfit nhất với ~85 episode |
| `dummy` | backbone + ACT tí hon, ngẫu nhiên | chỉ để kiểm pipeline trên CPU |

Lưu ý `frozen` với depth: gradient vẫn phải đi ngược qua toàn bộ ViT để tới
patch embedding (như smolvla_rgbd). Thiếu bộ nhớ thì
`finetune.gradient_checkpointing=true`.

### 3.4 Depth: nhánh riêng và distill

![ACT với nhánh depth và distill](images/architecture_depth.png)

Cách mặc định ở trên (depth là kênh thứ 4) có ba vấn đề, nêu trong
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): depth chỉ đi qua một lớp
khởi tạo 0 trước một backbone đóng băng chỉ biết RGB; điểm không đo được bị
coi là 0.2 m; depth lưu 8 bit rồi nén MP4. Repo có thêm:

| Cờ train | `data.depth_as` | Depth vào mạng thế nào | Lúc suy luận |
|---|---|---|---|
| `--depth-branch` | `branch` | CNN nhỏ (~0.6M, train từ đầu, lr 1e-4) đọc [depth, valid], ra lưới token của backbone RGB, **cộng** vào feature map RGB | như lúc train |
| `--lingbot-depth` | `branch` | LingBot-Depth ViT-L/14 (đóng băng) đọc RGB + depth, ra [1024, 15, 20], qua Conv1×1 (train) rồi cộng | thêm một ViT-L mỗi chunk |
| `--lingbot-depth-distill` | `branch` (hoặc `none`) | như `--depth-branch`; thêm loss phụ lúc train: token ảnh ở đầu ra encoder ACT, qua LayerNorm + Linear 512→1024, dự đoán đặc trưng của LingBot-Depth cùng patch; loss += 0.1 · (1 − cos) | thầy không chạy, không lưu trong checkpoint |

Những điểm thiết kế:

- **Dữ liệu** (`data=openarm_rgbd_branch`, `depth_format: raw`): depth thô
  lưu bằng feature depth của LeRobot (video HEVC 12 bit lượng tử log, không
  mất mát; đọc lại theo mét, sai số dưới 1 mm trong 0.2–1.2 m). Giá trị 0 đọc
  lại thành 0.01 m, nên mọi điểm < 0.1 m là "không đo được" (mask `valid`).
  Lúc serve, depth uint16 của client đi qua `images.depth_to_metres`: cùng
  tensor, cùng ngưỡng. Depth và mask resize nearest để không trộn điểm thiếu
  với điểm thật.
- **Khởi tạo 0:** lớp ra của nhánh depth (CNN hay Conv1×1 sau LingBot-Depth)
  bằng 0, nên lúc đầu model y hệt bản chỉ RGB và depth được học thêm vào.
- **Backbone RGB giữ nguyên 3 kênh**, không nới kênh: đặc trưng pretrain
  không bị xáo trộn.
- **Optimizer:** nhánh depth nằm trong `model.backbone.depth.*` nhưng học với
  lr của ACT (1e-4), không phải lr của backbone (1e-5); có test kiểm.
- **Distill** cần backbone RGB không học gì thêm (`frozen`) vẫn có tác dụng,
  vì loss tác động vào encoder Transformer của ACT (được train), không phải
  vào backbone. Thầy được tạo lần đầu khi train, đặt ngoài cây module
  (không vào `state_dict`, không vào optimizer). `data.depth_as=none` cho học
  sinh chỉ RGB: robot không cần camera depth lúc chạy.
- **LingBot-Depth** (Apache 2.0) được chép phần encoder vào
  `src/openarm_act_lerobot/lingbot_depth/` (xem `NOTICE.md`), so từng số với
  code gốc trên một frame thật: giống hệt. Lúc dùng phải tắt việc che ngẫu
  nhiên token depth (`enable_depth_mask=False`, mặc định của nó là che 60%).
- **Lỗi tên khoá trong lingbot-vla-v2:** commit 311a4d2 của Robbyant đổi tên
  lớp nhúng depth thành `depth_mask_patch_embed`, nhưng checkpoint
  `lingbot-depth-pretrain-vitl-14` (revision 8a4b168) lưu `depth_patch_embed`;
  nạp với `strict=False` thì lớp đó giữ trọng số ngẫu nhiên. Repo này nhận cả
  hai tên và nạp bắt buộc khớp. openarm_lingbot (bản 2.0) dùng đúng code đó:
  cần kiểm tra checkpoint thầy của nó có tên nào.

## 4. Dùng LeRobot thế nào

Như openarm_smolvla: không dùng CLI `lerobot-train`. `train_config.py` dựng
`OpenArmACTConfig` và `TrainPipelineConfig` từ Hydra, `run.py` gọi thẳng
`lerobot.scripts.lerobot_train.train(cfg)`, với bốn điểm chèn bằng
`mock.patch`:

1. **Norm stats** của repo thay stats per-frame (đúng cho cả delta actions).
2. **Delta actions** (`processors.py`, dùng chung với smolvla) khi
   `data.delta_actions=true`.
3. **Optimizer**: `use_policy_training_preset=false` để optimizer và lịch lr
   lấy từ Hydra, nhưng khi đó LeRobot đưa `policy.parameters()` thành một
   nhóm duy nhất. Repo thay bằng `policy.get_optim_params()` để backbone có
   lr riêng (test kiểm: hai nhóm 1e-4 / 1e-5).
4. **`openarm_config.yaml`** ghi vào mỗi checkpoint.

Siêu tham số: ACT của LeRobot (d_model 512, 8 head, ff 3200, encoder 4 lớp,
decoder 1 lớp — bản gốc 7 lớp nhưng bug chỉ dùng lớp đầu, CVAE 4 lớp, latent
32, KL 10, dropout 0.1). Optimizer theo act_pipeline: AdamW lr 1e-4, backbone
1e-5, weight decay 1e-4, cosine với 1000 bước warmup. Batch 32 × 40 000 bước
≈ 1.3M mẫu, cùng cỡ với công thức ACT của LeRobot (100k × 8); với ~70 episode
train × ~950 frame là ~19 epoch.

## 5. Dữ liệu

| ACT nhận | Từ OpenArm | Ghi chú |
|---|---|---|
| `observation.images.chest` | `chest_rgb` | float [3,H,W] 0..1, model resize + chuẩn hoá |
| `observation.images.chest_depth` | `chest_depth` | ảnh xám (`images.depth_to_image`), kênh 0 dùng làm kênh 4 |
| `observation.state` | `qpos` [16] | mean/std |
| `action` | nhãn [50,16] | tuyệt đối, `action[t:t+50]`, pad + mask cuối episode |

- `data=openarm_rgbd` (mặc định): đúng input/nhãn của act_pipeline.
- `data=openarm_rgb_depth`: depth là ảnh thứ hai qua cùng backbone (thêm
  ~300 token). `data=openarm_rgb`: chỉ RGB.
- ACT không đọc prompt; task vẫn được ghi vào dataset (LeRobot cần), episode
  không có task nhận `data.fallback_task`. Với cùng `data=`, dataset trùng với
  của openarm_smolvla và có thể dùng lại (`paths.lerobot_home=`).
- Depth được mã hoá 8 bit (bước ~4 mm) và nén video; act_pipeline đọc depth
  thô. Nếu depth quan trọng, thử `data.video=false`.

## 6. Checkpoint, release, serving

- Checkpoint theo layout LeRobot: `<step>/pretrained_model/` (trọng số,
  `config.json`, processors kèm stats, `openarm_config.yaml`) và
  `<step>/training_state/`.
- Release (`export_policy.py`) bỏ optimizer, tuỳ chọn bf16; tự chứa, serve
  không cần mạng.
- Server nói giao thức openpi (msgpack + NumPy, metadata khi kết nối,
  `GET /healthz`). Client `RemoteActPolicy` không cần prompt.
- ACT nhẹ và tất định nên robot có thể hỏi **mỗi bước** rồi để temporal
  ensembling của `openarm_act` trộn các chunk, như act_pipeline. Điều kiện:
  round trip < 20 ms. Trên CPU, model dummy mất ~37 ms/chunk; con số GPU với
  ViT-B cần đo ở `smoke_train.sh`.

## 7. Đánh giá

Offline eval trên các episode held-out, so với baseline "đứng yên"; không cần
cố định seed vì z = 0. Thí nghiệm chính, cùng dữ liệu, cùng tập val:

1. `backbone=resnet18` (baseline act_pipeline) vs `dinov2_vitb14` vs `dinov3_vitb16`, `finetune=frozen`.
2. Backbone tốt nhất với `finetune=partial` / `full`.
3. `data.depth_init=zero` vs `mean`; `data=openarm_rgb_depth` vs `openarm_rgbd`.

Trên robot: cùng task, cùng tư thế đầu, N = 20 lần mỗi model, so với
act_pipeline hiện tại, SmolVLA, π0.

## 8. Rủi ro và câu hỏi mở

- **Quyền DINOv3**: repo gated, Meta duyệt tay, có thể mất thời gian. Trong
  lúc chờ, DINOv2 và ResNet chạy được ngay.
- **Giấy phép**: DINOv3 dùng DINOv3 License; xem trước khi đăng release có
  trọng số DINOv3 lên Hub công khai.
- **Depth vào ViT đóng băng**: ViT chưa từng thấy kênh 4; chỉ patch embedding
  học được nó. Nếu depth không giúp, thử `partial`, hoặc `openarm_rgb_depth`.
- **Kích thước nhỏ của dữ liệu** (~85 episode, một task): `full` dễ overfit;
  ưu tiên `frozen`/`partial`.
- **Chạy trên robot không GPU**: có thể export ONNX tương thích `OnnxPolicy`
  của openarm_act (ảnh [1,4,H,W] chuẩn hoá ImageNet + depth 0..1, resize
  bilinear giống cv2 — repo đã giữ cách resize này). Chưa làm.
- **Chưa đo trên GPU**: bộ nhớ và tốc độ với batch 32, ViT-B, backward qua
  ViT để train patch embedding.

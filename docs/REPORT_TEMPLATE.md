# Mẫu báo cáo kết quả R0 / R1

Điền mẫu này sau khi train xong rồi gửi lại. Có hai phần:

- **Phần A — đo offline**, chạy trên máy GPU bằng repo này: số liệu trên tập val, kiểm tra hai run có so được với nhau không.
- **Phần B — thử trên robot**, chạy bằng repo khác, trên robot khác: thiết lập và kết quả từng lần thử.

Cách gửi nhanh nhất: dán nguyên output của các lệnh ở phần A, và bảng ở phần B (dạng markdown hoặc CSV).

---

## Phần A — đo offline (máy GPU, repo này)

Thư mục run mặc định: `checkpoints/act_dinov3_vitb16_frozen/r0_rgb` và
`.../r1_rgbd` (khác backbone hay finetune thì đổi tên thư mục cho đúng).

### A1. Thông tin train

| | R0 | R1 |
|---|---|---|
| Lệnh train đầy đủ | | |
| GPU (`nvidia-smi`) | | |
| Thời gian train | | |
| Step của checkpoint đánh giá | | |
| `l1_loss`, `kld_loss` ở cuối (dòng log `step:...` cuối, hoặc wandb) | | |
| Link wandb (nếu có) | | |

### A2. Offline eval

```bash
uv run scripts/offline_eval.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/r0_rgb
uv run scripts/offline_eval.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/r1_rgbd
```

Gửi: phần JSON lệnh in ra (model và hold), hoặc gửi kèm hai file
`offline_eval_val.json` nằm cạnh checkpoint.

### A3. Hai run có cùng tập val không

```bash
for d in openarm_rgb_qpos_50hz openarm_rgbd_qpos_50hz; do
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); v=sorted(s['id'] for s in r['skipped'] if s['reason'].startswith('split val')); print(sys.argv[1].split('/')[-2], len(v), v)" \
    data/lerobot/openarm/$d/openarm_conversion.json
done
```

Gửi: output của lệnh (hai danh sách phải giống nhau).

### A4. Trọng số kênh depth của R1

```bash
uv run scripts/inspect_checkpoint.py checkpoint=checkpoints/act_dinov3_vitb16_frozen/r1_rgbd
```

Gửi: các dòng lệnh in ra. Trọng số kênh depth bắt đầu từ 0; tỉ lệ
`depth / mean(RGB)` còn rất nhỏ sau khi train nghĩa là model gần như không
dùng depth. Với các run có nhánh depth (`--depth-branch`, `--lingbot-depth`,
`--lingbot-depth-distill`), lệnh in thêm độ lớn lớp ra của nhánh depth (bắt đầu từ 0).

---

## Phần B — thử trên robot (repo khác, robot khác)

### B0. Model được chạy trên robot thế nào

Đây là điều quan trọng nhất của phần B. Model chỉ cho kết quả đúng nếu ảnh và
state được xử lý **giống hệt lúc train**.

**Cách khuyến nghị:** chạy `scripts/serve.py` của repo này trên máy GPU, rồi
gọi từ robot bằng client `RemoteActPolicy` (`client/`, chỉ cần numpy, msgpack,
websockets). Mọi bước xử lý nằm ở server, nên không lệch được.

**Nếu repo kia tự nạp checkpoint và tự xử lý,** phải khớp đủ các điểm sau, và
ghi vào bảng B1 là đã khớp từng điểm chưa:

| Điểm | Lúc train |
|---|---|
| Thứ tự kênh màu | **RGB** (camera ROS hay trả BGR) |
| Kích thước khung | đúng kích thước của dataset (`image_size` trong metadata của server; data_openarm_8_10 là 640×480) |
| Resize vào model | bilinear, `align_corners=False`, tới `image_size` trong `config.json` của checkpoint |
| Chuẩn hoá RGB | ÷255, rồi mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225) |
| Depth (R1) | uint16 mm → m, kẹp 0.2–1.2 m, → 0..1, làm kênh thứ 4; **lượng tử hoá 8 bit như lúc convert** |
| qpos | 16 khớp đúng thứ tự `left_j1..j7, left_gripper, right_j1..j7, right_gripper`; rad, gripper theo hành trình ngón (m); chuẩn hoá bằng mean/std trong processors của checkpoint |
| Output | bỏ chuẩn hoá bằng mean/std của action; `[50, 16]` target tuyệt đối, hàng 0 là bước hiện tại |

### B1. Thiết lập

| | R0 | R1 |
|---|---|---|
| Robot, camera | | |
| Checkpoint hoặc release (tên, step) | | |
| Cách chạy model (server + `RemoteActPolicy`, hay repo khác tự nạp) | | |
| Nếu tự nạp: đã khớp đủ bảng B0 chưa (ghi điểm nào chưa khớp) | | |
| Độ phân giải camera lúc chạy | | |
| `control_rate`, `query_period`, `temporal_ensemble`, `ensemble_k` | | |
| `gripper_close_below` | | |
| Thời gian một lần suy luận (`server_ms` / round trip, hoặc tự đo) | | |

### B2. Cách thử

Để R0 và R1 so được với nhau:

- Cùng task, cùng tư thế đầu của tay, cùng bộ vị trí vật. Ví dụ 5 vị trí × 4
  lần = 20 lần mỗi model.
- Chạy **xen kẽ** R0 và R1 (R0, R1, R0, R1, ...) để ánh sáng hay độ mòn của
  robot ảnh hưởng đều cả hai.
- Ghi rõ tiêu chí thành công **trước khi** thử, ví dụ "hộp xanh nằm hẳn trên
  tờ giấy trắng, tay đã nhả ra".

### B3. Kết quả từng lần thử

| Run | Lần | Vị trí vật | Thành công (1/0) | Kiểu lỗi | Thời gian (s) | Ghi chú |
|---|---|---|---|---|---|---|
| R0 | 1 | A | | | | |
| R1 | 1 | A | | | | |
| ... | | | | | | |

Kiểu lỗi, chọn một:
`trượt khi gắp` · `rơi giữa đường` · `đặt sai chỗ` · `va chạm` · `đứng im / do dự` · `rung giật` · `quá thời gian` · `khác (ghi chú)`

### B4. Tổng hợp

| | R0 | R1 |
|---|---|---|
| Thành công / tổng số lần | | |
| Kiểu lỗi hay gặp nhất | | |
| Nhận xét định tính (mượt hay giật, có phản ứng khi dời vật không, ...) | | |

Nếu có: gửi kèm vài video ngắn của lần thành công và lần lỗi điển hình.

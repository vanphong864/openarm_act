"""Draws architecture_depth.svg: ACT with a depth branch and distillation.

    python3 docs/images/architecture_depth.py
    uvx --from cairosvg cairosvg docs/images/architecture_depth.svg -o docs/images/architecture_depth.png -s 1.5

Black on white: thin box = data, thick = trained, dashed = frozen,
dotted group = training only.
"""

from html import escape
from pathlib import Path

FONT = 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"'
STYLE = {"data": 'stroke-width="1.75"', "trained": 'stroke-width="3.5"', "frozen": 'stroke-width="1.75" stroke-dasharray="7 4"'}


class Svg:
    def __init__(self, width, height):
        self.parts = [
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
            '<path d="M 0 1 L 8 4.5 L 0 8 Z" fill="#000"/></marker></defs>',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        ]

    def text(self, x, y, s, size=14, bold=False, anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{700 if bold else 400}" '
                          f'fill="#000" text-anchor="{anchor}" dominant-baseline="central">{escape(s)}</text>')

    def box(self, x, y, w, h, lines, kind="data"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" stroke="#000" {STYLE[kind]}/>')
        top = y + h / 2 - 9.5 * (len(lines) - 1)
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + 19 * i, line, bold=i == 0 and len(lines) > 1)

    def group(self, x, y, w, h, label):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="#000" '
                          'stroke-width="1.25" stroke-dasharray="2 4"/>')
        self.text(x + 14, y + 18, label, size=13, bold=True, anchor="start")

    def arrow(self, *points, label=None, at=None):
        d = "M " + " L ".join(f"{x} {y}" for x, y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1.75" marker-end="url(#a)"/>')
        if label:
            self.text(*at, label, size=12.5, anchor="start")

    def plus(self, cx, cy, r=20):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="#000" stroke-width="1.75"/>')
        self.text(cx, cy, "+", size=22, bold=True)

    def save(self, path):
        Path(path).write_text("\n".join(self.parts + ["</svg>", ""]))


s = Svg(1420, 1220)
s.text(24, 28, "ACT với nhánh depth và distill  (--depth-branch, --lingbot-depth, --lingbot-depth-distill)",
       size=18, bold=True, anchor="start")
for x, kind, label in ((24, "data", "dữ liệu"), (130, "trained", "được train"), (270, "frozen", "đóng băng (pretrained)")):
    s.box(x, 46, 34, 18, [], kind)
    s.text(x + 42, 55, label, size=13, anchor="start")
s.group(470, 46, 34, 18, "")
s.text(512, 55, "chỉ lúc train", size=13, anchor="start")
s.text(640, 55, "shape mỗi mẫu (không có chiều batch); khung 640×480, backbone DINOv3 ViT-B/16", size=13, anchor="start")

# ------------------------------------------------------------------ inputs
s.box(30, 80, 300, 52, ["chest RGB", "uint8 480×640×3"])
s.box(370, 80, 300, 52, ["chest depth", "uint16 480×640 mm (0 = không đo)"])
s.box(1090, 80, 145, 52, ["qpos", "float32 [16]"])
s.box(1245, 80, 145, 52, ["nhãn action", "[50, 16], khi train"])
s.box(30, 160, 300, 56, ["÷255, resize bilinear", "[3, 240, 320]"])
s.box(370, 160, 300, 56, ["mét; valid = depth > 0.1 m", "nearest → [2, 240, 320]"])
s.box(1090, 160, 145, 56, ["chuẩn hoá", "mean / std"])
s.box(1245, 160, 145, 56, ["CVAE → z [32]", "suy luận: z = 0"], "trained")
for x in (180, 520, 1162, 1317):
    s.arrow((x, 132), (x, 158))

# ------------------------------------------------------------------ vision
s.box(30, 262, 300, 84, ["backbone RGB", "--resnet | --dinov2 | --dinov3", "chuẩn hoá ImageNet, 3 kênh",
                         "→ [C, 15, 20], C = 768"], "frozen")
s.arrow((180, 216), (180, 260))
s.text(705, 246, "depth encoder: một trong hai", size=13, bold=True)
s.box(370, 262, 300, 84, ["CNN nhỏ  (--depth-branch)", "4 × conv stride 2 + GroupNorm",
                          "pool về 15×20, Conv1×1 → C", "lớp cuối khởi tạo 0"], "trained")
s.arrow((520, 216), (520, 260))
s.box(710, 262, 340, 84, ["LingBot-Depth ViT-L/14  (--lingbot-depth)", "đọc RGB + depth cùng lúc",
                          "→ [1024, 15, 20]", "đóng băng, chạy cả lúc suy luận"], "frozen")
s.arrow((670, 188), (880, 188), (880, 260))
s.text(688, 176, "RGB + depth", size=12.5, anchor="start")
s.box(710, 372, 340, 48, ["Conv1×1 1024→C", "khởi tạo 0"], "trained")
s.arrow((880, 346), (880, 370))
s.plus(520, 470)
s.arrow((180, 346), (180, 470), (498, 470))
s.arrow((520, 346), (520, 448))
s.arrow((880, 420), (880, 470), (542, 470))
s.text(560, 445, "cộng theo từng patch", size=12.5, anchor="start")
s.box(370, 516, 300, 56, ["conv 1×1 C→512 + 2D sin", "300 token ảnh [300, 512]"], "trained")
s.arrow((520, 492), (520, 514))

# ------------------------------------------------------------ state, latent
s.box(1090, 262, 145, 56, ["Linear 16→512", "token state"], "trained")
s.box(1245, 262, 145, 56, ["Linear 32→512", "token latent"], "trained")
s.arrow((1162, 216), (1162, 260))
s.arrow((1317, 216), (1317, 260))

# ----------------------------------------------------------------- encoder
s.box(30, 620, 1360, 48, ["đầu vào encoder: [latent, state, 300 token ảnh] = 302 token × 512"])
s.arrow((520, 572), (520, 618))
s.arrow((1162, 318), (1162, 618))
s.arrow((1317, 318), (1317, 618))
s.box(30, 696, 1360, 48, ["Transformer encoder: 4 lớp, 8 head, FFN 3200", "memory [302, 512]"], "trained")
s.arrow((710, 668), (710, 694))

# ----------------------------------------------------------------- decoder
s.box(30, 790, 280, 56, ["50 query (0)", "+ vị trí học được [50, 512]"])
s.box(340, 790, 540, 56, ["Transformer decoder: 1 lớp", "self-attention 50 query, cross-attention 302 token memory"], "trained")
s.arrow((310, 818), (338, 818))
s.arrow((610, 744), (610, 788))
s.box(340, 874, 540, 48, ["Linear 512→16", "[50, 16] đã chuẩn hoá"], "trained")
s.arrow((610, 846), (610, 872))
s.box(340, 950, 540, 48, ["bỏ chuẩn hoá (mean / std của action)", "(+ state nếu data.delta_actions)"])
s.arrow((610, 922), (610, 948))
s.box(340, 1026, 540, 52, ["actions float32 [50, 16]", "target khớp tuyệt đối; hàng 0 = bước hiện tại"])
s.arrow((610, 998), (610, 1024))

# ------------------------------------------------------------------ distill
s.group(920, 772, 470, 344, "chỉ lúc train: --lingbot-depth-distill")
s.box(1180, 806, 195, 56, ["300 token ảnh", "từ encoder [300, 512]"])
s.arrow((1277, 744), (1277, 804))
s.box(1180, 890, 195, 56, ["distill head", "LN + Linear 512→1024"], "trained")
s.arrow((1277, 862), (1277, 888))
s.box(935, 806, 225, 140, ["LingBot-Depth (thầy)", "đọc RGB + depth", "→ [300, 1024]", "đóng băng,",
                           "không lưu vào checkpoint"], "frozen")
s.box(935, 974, 440, 48, ["loss_distill = 1 − cos(dự đoán, thầy)", "trung bình trên 300 patch"])
s.arrow((1047, 946), (1047, 972))
s.arrow((1277, 946), (1277, 972))
s.box(935, 1050, 440, 48, ["loss = L1 + 10 · KL + 0.1 · loss_distill", "distill.weight = 0.1"])
s.arrow((1155, 1022), (1155, 1048))

# ------------------------------------------------------------------- notes
notes = [
    "Không có cờ depth: depth là kênh thứ 4 của backbone (architecture.png). --lingbot-depth và --lingbot-depth-distill không dùng cùng lúc.",
    "Cờ backbone: --resnet (ResNet-18, ảnh 640×480 giữ nguyên, stride 32) · --dinov2 (ViT-B/14, 280×210) · --dinov3 (ViT-B/16, 320×240, mặc định).",
    "Distill cho học sinh RGB-only (robot không cần depth): thêm data.depth_as=none. Dữ liệu: data=openarm_rgbd_branch (depth thô, 12 bit, không mất mát).",
]
for i, note in enumerate(notes):
    s.text(24, 1146 + 22 * i, note, size=13, anchor="start")

s.save(Path(__file__).with_name("architecture_depth.svg"))

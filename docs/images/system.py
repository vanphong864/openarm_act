"""Draws system.svg: from recording to the robot, with the openarm_act1 node.

    python3 docs/images/system.py
    uvx --from cairosvg cairosvg docs/images/system.svg -o docs/images/system.png -s 1.5

Black on white, as architecture_depth.py.
"""

from html import escape
from pathlib import Path

FONT = 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"'


class Svg:
    def __init__(self, width, height):
        self.parts = [
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto-start-reverse">'
            '<path d="M 0 1 L 8 4.5 L 0 8 Z" fill="#000"/></marker></defs>',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        ]

    def text(self, x, y, s, size=14, bold=False, anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{700 if bold else 400}" '
                          f'fill="#000" text-anchor="{anchor}" dominant-baseline="central">{escape(s)}</text>')

    def box(self, x, y, w, h, lines, thick=False):
        width = 3.5 if thick else 1.75
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" stroke="#000" '
                          f'stroke-width="{width}"/>')
        top = y + h / 2 - 9.5 * (len(lines) - 1)
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + 19 * i, line, bold=i == 0 and len(lines) > 1)

    def group(self, x, y, w, h, label):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="#000" '
                          'stroke-width="1.25" stroke-dasharray="3 4"/>')
        self.text(x + 14, y + 20, label, size=14, bold=True, anchor="start")

    def arrow(self, *points, both=False):
        d = "M " + " L ".join(f"{x} {y}" for x, y in points)
        start = ' marker-start="url(#a)"' if both else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1.75" '
                          f'marker-end="url(#a)"{start}/>')

    def save(self, path):
        Path(path).write_text("\n".join(self.parts + ["</svg>", ""]))


s = Svg(1240, 480)
s.text(20, 28, "openarm_act: from recording to the robot", size=18, bold=True, anchor="start")
s.group(20, 50, 450, 410, "Robot PC  (ROS 2 Humble, Python 3.10, no GPU)")
s.group(790, 50, 430, 410, "GPU machine  (Python 3.12, this repo)")

s.box(40, 96, 410, 56, ["openarm_mc", "teleop + recorder"])
s.box(40, 190, 410, 56, ["openarm_ui: the ACT1 button", "Generate · Start · Execute · Stop"])
s.box(40, 290, 410, 76, ["openarm_act1 node (new)", "openarm_act's node, inherited, not changed",
                         "joint-state history at 50 Hz → the last 10 qpos"], thick=True)
s.text(245, 400, "openarm_act node: unchanged, for the others' models", size=13)
s.text(245, 420, "command proxy source act1 → the arm", size=13)
s.arrow((245, 246), (245, 288))

s.box(520, 96, 220, 56, ["Hugging Face dataset", "HDF5 episodes"])
s.box(810, 96, 390, 56, ["run_pipeline.sh · train_overnight.sh", "download → convert → norm stats → train"])
s.box(810, 190, 390, 44, ["checkpoint → release (export_policy.py)"])
s.box(810, 290, 390, 76, ["scripts/serve.py", "websocket :8000", "same preprocessing as training"])

s.arrow((450, 124), (518, 124))
s.text(484, 108, "HDF5", size=13)
s.arrow((740, 124), (808, 124))
s.arrow((1005, 152), (1005, 188))
s.arrow((1005, 234), (1005, 288))
s.arrow((450, 328), (808, 328), both=True)
s.text(629, 312, "rgb, depth, last 10 qpos  →", size=13)
s.text(629, 344, "←  actions [50, 16]", size=13)

s.save(Path(__file__).with_name("system.svg"))

"""Cảnh MuJoCo tái tạo setup OpenArm v1.0 hai tay + hộp kê + camera ngực D435i."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "scene_openarm.yaml"
DEFAULT_SCENE = ROOT / "scenes" / "openarm.xml"

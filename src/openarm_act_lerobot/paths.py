"""Where things live, and the environment LeRobot reads at import.

Relative paths in the Hydra configs are relative to the repository, not to
wherever a script was started from.
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"
# Written into every checkpoint's pretrained_model/ and every release: the
# resolved Hydra config that serve.py, offline_eval.py and export_policy.py
# rebuild the data pipeline from.
RUN_CONFIG = "openarm_config.yaml"


def resolve(path: str | Path) -> Path:
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else REPO_ROOT / path


# KEY=value lines the scripts read at start: WANDB_API_KEY, HF_TOKEN, ...
DOTENV = REPO_ROOT / ".env"


def load_dotenv(path: Path = DOTENV) -> list[str]:
    """Set the variables of a .env file; -> their names. A variable already
    in the environment wins; empty values are skipped."""
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def setup_env(cfg: DictConfig) -> None:
    """Load .env, and point LeRobot at this repo's datasets.

    Must run before lerobot is imported: it reads HF_LEROBOT_HOME once, at
    import. An environment variable the user set wins. The Hub stays
    reachable: the ViT backbones' weights come from there.
    """
    load_dotenv()
    os.environ.setdefault("HF_LEROBOT_HOME", str(resolve(cfg.paths.lerobot_home)))

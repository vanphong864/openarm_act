#!/usr/bin/env bash
# Full pipeline on a GPU machine: environment, backbone weights, Hub episodes, LeRobot
# dataset, norm stats, ACT training. Safe to run again: each step does only what
# is missing (see scripts/train_pipeline.py), and resume=true continues a run.
#
#   scripts/run_pipeline.sh exp_name=v1
#   scripts/run_pipeline.sh exp_name=v1 batch_size=32
#   scripts/run_pipeline.sh exp_name=v1 resume=true
#   scripts/run_pipeline.sh exp_name=v1 backbone=resnet18
#   scripts/run_pipeline.sh exp_name=v1 backbone=dinov2_vitb14 finetune=partial
#
# Arguments are Hydra overrides for conf/pipeline.yaml (= train.yaml + download).
set -euo pipefail
cd "$(dirname "$0")/.."

# uv: installed for this user (~/.local/bin, no sudo) when missing. It then
# fetches Python 3.12 itself, so the machine needs no Python of its own.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo "installing uv into ~/.local/bin" >&2
    if command -v curl >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        wget -qO- https://astral.sh/uv/install.sh | sh
    fi
fi
uv --version
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "warning: no nvidia-smi; training would run on the CPU" >&2
fi
if ! command -v ffmpeg >/dev/null; then
    echo "note: no ffmpeg; videos decode with PyAV (slower). sudo apt install ffmpeg to use torchcodec" >&2
fi
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "created .env: paste WANDB_API_KEY into it (or pass wandb.enable=false)" >&2
fi

# ROS puts its Python 3.10 packages on PYTHONPATH; this environment is Python 3.12.
unset PYTHONPATH
uv sync
uv run python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU seen:', torch.cuda.is_available())"
# A GPU torch cannot use (a driver older than its CUDA wants) would make
# device=auto train on the CPU, for days: stop instead.
if command -v nvidia-smi >/dev/null && ! uv run python -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    echo "error: there is a GPU but torch cannot use it. NVIDIA driver here: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1);" >&2
    echo "this torch is built for CUDA $(uv run python -c 'import torch; print(torch.version.cuda)'), which needs a newer driver." >&2
    echo "Update the driver, or pass device=cpu if CPU training is really meant." >&2
    if [[ " $* " != *" device=cpu "* ]]; then exit 1; fi
fi

uv run scripts/train_pipeline.py "$@"

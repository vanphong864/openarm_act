#!/usr/bin/env bash
# A short run of everything, on the GPU, before the long one: the real
# backbone and data config, 3 episodes, a few dozen steps. Checks the setup,
# access to the backbone (DINOv3's is gated), downloading, conversion, training at the
# real batch size (an out-of-memory shows here), checkpoints, resume,
# offline evaluation, export, and serving a real frame through the robot's client.
#
#   scripts/smoke_train.sh
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 scripts/smoke_train.sh
#   scripts/smoke_train.sh batch_size=16          # extra Hydra overrides
#   scripts/smoke_train.sh --resnet               # the same flags/overrides as the real run
#   scripts/smoke_train.sh --lingbot-depth
#   scripts/smoke_train.sh wandb.enable=false     # without a wandb key
#
# Its outputs are named "smoke" and stay apart from real runs (wandb project
# openarm_act_smoke); the episodes it downloads are reused by the real run.
# Takes ~10-20 min, mostly the first `uv sync` and downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP=smoke
PORT="${SMOKE_PORT:-8765}"
# The same for every step, plus whatever is passed on the command line.
COMMON=(
    data.name=smoke data.max_episodes=3 exp_name="$EXP" wandb.project=openarm_act_smoke
    log_interval=5 save_interval=10 lr.warmup_steps=5 "$@"
)
RELEASE="releases/$EXP"
started=$(date +%s)

step() { printf '\n\033[1m##### %s\033[0m\n' "$*"; }

step "1/6 setup, backbone, download, convert, norm stats, train 20 steps"
scripts/run_pipeline.sh "${COMMON[@]}" download.limit=3 num_train_steps=20 overwrite=true

export PATH="$HOME/.local/bin:$PATH"
unset PYTHONPATH
# The run directory the overrides lead to (backbone= and finetune= name it).
CHECKPOINT=$(uv run python - "${COMMON[@]}" <<'PY'
import sys
from hydra import compose, initialize_config_dir
from openarm_act_lerobot import paths
from openarm_act_lerobot.cli import expand_flags
with initialize_config_dir(version_base="1.3", config_dir=str(paths.CONF_DIR)):
    cfg = compose(config_name="train", overrides=expand_flags(["smoke", *sys.argv[1:]])[1:])
print(f"{cfg.paths.checkpoints}/{cfg.name}/{cfg.exp_name}")
PY
)

step "2/6 resume to 30 steps"
uv run scripts/train.py "${COMMON[@]}" num_train_steps=30 resume=true
ls "$CHECKPOINT/checkpoints"

step "3/6 offline evaluation on the held-out episode"
uv run scripts/offline_eval.py checkpoint="$CHECKPOINT" every=50

step "4/6 export"
uv run scripts/export_policy.py checkpoint="$CHECKPOINT" name="$EXP" overwrite=true
du -sh "$RELEASE"

step "5/6 serve, and query it with the robot's client"
mkdir -p outputs
HF_HUB_OFFLINE=1 uv run scripts/serve.py checkpoint="$RELEASE" port="$PORT" > outputs/smoke_serve.log 2>&1 &
server=$!
stop_server() { kill "$server" 2>/dev/null || true; pkill -f "scripts/serve.py checkpoint=$RELEASE" 2>/dev/null || true; }
trap stop_server EXIT
uv run python - "$PORT" <<'EOF_PY'
import glob, sys, time

import h5py
import numpy as np
from openarm_act_lerobot_client.policy import RemoteActPolicy

port = int(sys.argv[1])
deadline = time.time() + 600
while True:  # the server listens once the model is loaded
    try:
        policy = RemoteActPolicy(checkpoint=f"ws://127.0.0.1:{port}")
        break
    except ConnectionError:
        if time.time() > deadline:
            sys.exit("the server did not come up; see outputs/smoke_serve.log")
        time.sleep(2)
print(policy.describe(), "| frames", policy.image_size)

episode = sorted(glob.glob("data/raw/data_openarm_8_10/*.hdf5"))[0]
with h5py.File(episode) as f:
    rgb = f["observations/images/chest_rgb"][100]
    depth = f["observations/images/chest_depth"][100]
    qpos = f["observations/qpos"][100]
for _ in range(5):  # the first call warms up
    actions = policy.predict(rgb, depth, qpos)
assert actions.shape == (policy.chunk_size, 16) and np.isfinite(actions).all(), actions.shape
print(f"actions {actions.shape}, server {policy.server_ms:.0f} ms, round trip {policy.round_trip_ms:.0f} ms")
print("first target - qpos, arm joints (rad):", np.round(actions[0, :7] - qpos[:7], 3))
EOF_PY
curl -fsS "http://127.0.0.1:$PORT/healthz" || true

step "6/6 done in $(( ($(date +%s) - started) / 60 )) min"
cat <<EOF_MSG
Everything ran. Before the real training:
  - wandb: project openarm_act_smoke shows run $(basename "$(dirname "$CHECKPOINT")")_$EXP with l1_loss and kld_loss logged
  - GPU memory: mem_gb in the step lines of the training log above (the real batch size)
  - the server's time per chunk above: with temporal ensembling openarm_act asks for one every step (20 ms at 50 Hz)
Then:
  scripts/run_pipeline.sh exp_name=v1
Clean up the smoke run (keeps the downloaded episodes and the backbone):
  rm -rf $CHECKPOINT $RELEASE data/lerobot/openarm/smoke_* assets/*/openarm/smoke_*
EOF_MSG

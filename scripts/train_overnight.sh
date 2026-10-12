#!/usr/bin/env bash
# Four models overnight, ready to pick up in the morning:
#
#   resnet           --resnet                     ACT + ResNet-18 (act_pipeline's backbone), RGB-D 4th channel
#   dino             --dinov3                     ACT + DINOv3 ViT-B/16, RGB-D 4th channel
#   resnet_lingbot   --resnet --lingbot-depth     ResNet-18 + LingBot-Depth depth branch
#   dino_lingbot     --dinov3 --lingbot-depth     DINOv3 + LingBot-Depth depth branch
#
#   scripts/train_overnight.sh                          # one GPU, the four one after another
#   GPUS=0,1,2,3 scripts/train_overnight.sh             # four GPUs, one model each, side by side
#   DINO=dinov2 scripts/train_overnight.sh              # DINOv2 ViT-B/14 (no gated access needed)
#   LINGBOT=--lingbot-depth-distill scripts/train_overnight.sh   # LingBot-Depth as teacher instead
#   STEPS=20000 scripts/train_overnight.sh batch_size=16         # fewer steps; extra overrides reach every run
#
# 1. Prepares every run first (environment, backbones, episodes, datasets,
#    norm stats): a refused DINOv3 access or a full disk shows in the first
#    minutes, not in the morning. A run that cannot be prepared is skipped.
# 2. Trains each model, then evaluates it on the held-out episodes and
#    exports a release. One failing does not stop the others.
# 3. Writes outputs/overnight/<TAG>/summary.md: status, time, arm RMSE
#    against holding still, and the release of each model.
#
# Run it again with the same TAG to continue after an interruption: finished
# runs are skipped, unfinished ones resume from their last checkpoint.
# Start it inside tmux (tmux new -s night; Ctrl+B D to leave it running).
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-night_$(date +%Y%m%d)}"
DINO="${DINO:-dinov3}"                  # dinov3 | dinov2
LINGBOT="${LINGBOT:---lingbot-depth}"   # --lingbot-depth | --lingbot-depth-distill
STEPS="${STEPS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}" # 4 checkpoints a run; every one is kept
GPUS="${GPUS:-0}"
RUNS=(
    "resnet|--resnet"
    "dino|--$DINO"
    "resnet_lingbot|--resnet $LINGBOT"
    "dino_lingbot|--$DINO $LINGBOT"
)
COMMON=(exp_name="$TAG" num_train_steps="$STEPS" save_interval="$SAVE_INTERVAL" "$@")
LOGS="outputs/overnight/$TAG"
STATUS="$LOGS/status.tsv"
mkdir -p "$LOGS"
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n\033[1m##### %s\033[0m\n' "$*"; }
now() { date +%s; }

# checkpoints/<name>/<exp_name> of a run, from its flags and overrides.
run_dir() {
    env -u PYTHONPATH uv run --quiet python - "$@" <<'PY'
import sys
from hydra import compose, initialize_config_dir
from openarm_act_lerobot import paths
from openarm_act_lerobot.cli import expand_flags
with initialize_config_dir(version_base="1.3", config_dir=str(paths.CONF_DIR)):
    cfg = compose(config_name="pipeline", overrides=expand_flags(["run_dir", *sys.argv[1:]])[1:])
print(paths.resolve(cfg.paths.checkpoints) / cfg.name / cfg.exp_name)
PY
}

record() { printf '%s\t%s\t%s\t%s\t%s\n' "$@" >> "$STATUS"; }  # name, status, minutes, run dir, release

# ------------------------------------------------------------------ prepare
say "overnight $TAG: ${#RUNS[@]} models, $STEPS steps each, GPUs $GPUS; logs in $LOGS"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
echo "free disk: ${free_gb} GB (budget ~60 GB: datasets ~15, checkpoints ~2 GB each with LingBot-Depth)"
if (( free_gb < 60 )); then
    echo "warning: under 60 GB free; checkpoints may fill the disk overnight (SAVE_INTERVAL=$STEPS keeps only the last)" >&2
fi

: > "$STATUS"
READY=()
for run in "${RUNS[@]}"; do
    name="${run%%|*}"; flags="${run#*|}"
    say "prepare $name ($flags)"
    # shellcheck disable=SC2086
    if scripts/run_pipeline.sh $flags "${COMMON[@]}" prepare_only=true 2>&1 | tee "$LOGS/prepare_$name.log"; then
        READY+=("$run")
    else
        echo "!!! $name could not be prepared; it is skipped (see $LOGS/prepare_$name.log)" >&2
        record "$name" "not prepared" 0 "-" "-"
    fi
done
if (( ${#READY[@]} == 0 )); then
    echo "nothing to train" >&2
    exit 1
fi
say "prepared: ${#READY[@]} of ${#RUNS[@]}; training starts now, it is safe to leave"

# -------------------------------------------------------------------- train
train_one() {  # name, flags, gpu
    local name="$1" flags="$2" gpu="$3" started dir label mode release log
    started=$(now)
    log="$LOGS/train_$name.log"
    # shellcheck disable=SC2086
    if ! dir=$(run_dir $flags "${COMMON[@]}" 2>> "$log") || [[ -z "$dir" ]]; then
        echo "[$name] !!! its overrides do not compose, see $log" >&2
        record "$name" "bad overrides" 0 "-" "-"
        return
    fi
    label=$(printf '%06d' "$STEPS")
    mode=()
    if [[ -d "$dir/checkpoints/$label" ]]; then
        echo "[$name] already trained to $STEPS steps: $dir"
    else
        if [[ -e "$dir/checkpoints/last" ]]; then
            mode=(resume=true); echo "[$name] resuming $dir"
        elif [[ -d "$dir" ]]; then
            mode=(overwrite=true); echo "[$name] $dir has no checkpoint yet: starting over"
        fi
        echo "[$name] training on GPU $gpu -> $log"
        # shellcheck disable=SC2086
        if ! CUDA_VISIBLE_DEVICES="$gpu" env -u PYTHONPATH uv run scripts/train.py $flags "${COMMON[@]}" "${mode[@]}" \
                > "$log" 2>&1; then
            echo "[$name] !!! training failed, see $log" >&2
            record "$name" "training failed" $(( ($(now) - started) / 60 )) "$dir" "-"
            return
        fi
    fi
    echo "[$name] offline evaluation"
    CUDA_VISIBLE_DEVICES="$gpu" env -u PYTHONPATH uv run scripts/offline_eval.py checkpoint="$dir" >> "$log" 2>&1 \
        || echo "[$name] offline evaluation failed, see $log" >&2
    echo "[$name] export"
    release="releases/${name}_${TAG}"
    if env -u PYTHONPATH uv run scripts/export_policy.py checkpoint="$dir" name="${name}_${TAG}" overwrite=true \
            >> "$log" 2>&1; then
        record "$name" "done" $(( ($(now) - started) / 60 )) "$dir" "$release"
    else
        record "$name" "export failed" $(( ($(now) - started) / 60 )) "$dir" "-"
    fi
    echo "[$name] finished in $(( ($(now) - started) / 60 )) min"
}

IFS=',' read -r -a gpu_list <<< "$GPUS"
for (( q = 0; q < ${#gpu_list[@]}; q++ )); do
    (   # one queue per GPU: runs q, q + n, q + 2n, ... one after another
        for (( i = q; i < ${#READY[@]}; i += ${#gpu_list[@]} )); do
            run="${READY[$i]}"
            train_one "${run%%|*}" "${run#*|}" "${gpu_list[$q]}"
        done
    ) &
done
wait

# ------------------------------------------------------------------ summary
say "summary"
env -u PYTHONPATH uv run --quiet python - "$STATUS" "$LOGS/summary.md" "$TAG" <<'PY'
import json, sys
from pathlib import Path

status, out, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = [line.split("\t") for line in status.read_text().splitlines() if line.strip()]
lines = [f"# Overnight {tag}", "",
         "| Model | Status | Minutes | Arm RMSE (rad) | Hold still | Gripper MAE (mm) | Release |",
         "|---|---|---|---|---|---|---|"]
for name, state, minutes, run, release in rows:
    model = hold = gripper = "-"
    reports = sorted(Path(run).glob("checkpoints/*/offline_eval_val.json")) if run != "-" else []
    if reports:
        report = json.loads(reports[-1].read_text())
        model = f"{report['model']['arm_rmse_rad']:.4f}"
        hold = f"{report['hold']['arm_rmse_rad']:.4f}"
        gripper = f"{report['model']['gripper_mae_mm']:.1f}"
    lines.append(f"| {name} | {state} | {minutes} | {model} | {hold} | {gripper} | {release} |")
lines += ["", "A model is useful only if its arm RMSE is clearly below holding still.",
          "Serve one: uv run scripts/serve.py checkpoint=<release>"]
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
PY
echo "full logs: $LOGS"

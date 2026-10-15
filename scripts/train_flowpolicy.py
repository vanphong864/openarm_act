#!/usr/bin/env python3
"""Train upstream FlowPolicy on OpenArm XYZ/qpos sequences, save an inference release."""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from openarm_flow.data import FlowDataset
from openarm_flow.policy import build_model, save_release
from openarm_flow.plotting import plot_losses
from openarm_flow._vendor.model.flow.ema_model import EMAModel


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device)


@torch.no_grad()
def validate(model, loader, device, max_batches, seed):
    if not len(loader):
        return {}
    losses, errors, holds = [], [], []
    model.eval()
    # Fixed validation noise, without changing the training RNG stream.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.startswith("cuda") else []):
        torch.manual_seed(seed)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            losses.append(float(model.compute_loss(batch)[0]))
            prediction = model.predict_action(batch["obs"])["action"]
            first = model.n_obs_steps - 1
            target = batch["action"][:, first:first + model.n_action_steps]
            hold = batch["obs"]["agent_pos"][:, -1:, :].expand_as(target)
            errors.append((prediction - target).cpu().numpy())
            holds.append((hold - target).cpu().numpy())
    result = {"val_loss": float(np.mean(losses))}
    for name, values in [("policy", errors), ("hold", holds)]:
        e = np.concatenate(values)
        result[f"{name}_arm_rmse_rad"] = float(np.sqrt(np.mean(np.delete(e, [7, 15], axis=-1)**2)))
        result[f"{name}_gripper_mae_mm"] = float(np.mean(np.abs(e[..., [7, 15]])) * 1000)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=root / "conf/flowpolicy.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tr = cfg["train"]
    if args.steps is not None:
        tr["steps"] = args.steps
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if min(tr["steps"], tr["batch_size"], tr["save_every"], tr["val_batches"]) < 1:
        ap.error("steps, batch_size, save_every and val_batches must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume or a new output directory")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    model_cfg = cfg["model"]
    train = FlowDataset(args.dataset, model_cfg["horizon"], model_cfg["n_obs_steps"], "train")
    val = FlowDataset(args.dataset, model_cfg["horizon"], model_cfg["n_obs_steps"], "val")
    if not len(train):
        raise ValueError("Empty training split")
    model = build_model(model_cfg, train.metadata)
    state = None
    start = 0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
        if state["model_config"] != model_cfg or state["metadata"] != train.metadata:
            raise ValueError("Resume model/dataset metadata differs from checkpoint")
        model.load_state_dict(state["model"])
        start = state["step"]
    else:
        model.set_normalizer(train.normalizer())
    model.to(device)
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model, power=0.75)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tr["lr"],
                                 betas=(0.95, 0.999), eps=1e-8, weight_decay=tr["weight_decay"])
    if state:
        ema_model.load_state_dict(state["ema"])
        ema.optimization_step = start
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if device.startswith("cuda") and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    loader = DataLoader(train, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"])
    val_loader = DataLoader(val, batch_size=tr["batch_size"], shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train_config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"FlowPolicy on {device}: train={len(train)}, val={len(val)}, start={start}, steps={tr['steps']}", flush=True)
    iterator = iter(loader)
    for step in range(start + 1, tr["steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        warmup = tr["warmup_steps"]
        fraction = max(0, (step - warmup) / max(1, tr["steps"] - warmup))
        factor = min(1, step / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = tr["lr"] * factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(to_device(batch, device))
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        ema.step(model)
        if step == 1 or step % 100 == 0:
            print(f"step {step}: loss={float(loss.detach()):.8f}", flush=True)
            with (args.output / "train_loss.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "train_loss": float(loss.detach())}) + "\n")
        if step % tr["save_every"] == 0 or step == tr["steps"]:
            metrics = {"step": step, "train_loss": float(loss.detach()),
                       **validate(ema_model, val_loader, device, tr["val_batches"], cfg["seed"])}
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            save_release(args.output / "release", ema_model, model_cfg, train.metadata, step)
            state = {"step": step, "model_config": model_cfg, "metadata": train.metadata,
                     "model": model.state_dict(), "ema": ema_model.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
            torch.save(state, args.output / "last.pt.tmp")
            (args.output / "last.pt.tmp").replace(args.output / "last.pt")
            try:
                plot_losses(args.output)
            except Exception as exc:
                print(f"Could not update loss_curve.png: {exc}", flush=True)
    print(f"Release: {args.output / 'release'}")


if __name__ == "__main__":
    main()

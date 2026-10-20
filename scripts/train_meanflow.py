#!/usr/bin/env python3
"""Train dual-depth MeanFlow, validate held-out episodes and export last/best EMA."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
import yaml

from openarm_flow.meanflow import MeanFlowPolicy
from openarm_flow.meanflow_data import MeanFlowDataset
from openarm_flow.meanflow_policy import save_meanflow_release
from openarm_flow.plotting import plot_losses, plot_meanflow_metrics


def to_device(batch, device):
    return {k: to_device(v, device) for k, v in batch.items()} if isinstance(batch, dict) else batch.to(device)


@torch.no_grad()
def validate(model, loader, device, seed):
    model.eval()
    sums = {k: 0.0 for k in ("loss", "normalized_mse", "policy_arm", "policy_gripper", "hold_arm", "hold_gripper")}
    count, frames = 0, 0
    tcp_squares, tcp_angle_squares, tcp_count = 0.0, 0.0, 0
    approach_squares, approach_angle_squares, approach_grip_sum, approach_count = 0.0, 0.0, 0.0, 0
    approach_distances = []
    arms = [i for i in range(16) if i not in (7, 15)]
    cuda_devices = list(range(torch.cuda.device_count())) if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for batch in loader:
            batch = to_device(batch, device)
            loss, _ = model.compute_loss(batch)  # JVP locally enables grad; no parameter gradients retained
            prediction = model.predict_action(batch["obs"])["action"]
            target = batch["action"][:, :model.n_action_steps]
            mask = batch["action_valid"][:, :model.n_action_steps]
            hold = batch["obs"]["agent_pos"][:, -1:, :].expand_as(target)
            error = (prediction - target)[mask]
            sums["normalized_mse"] += float((error / model.action_scale).square().sum())
            for name, candidate in (("policy", prediction), ("hold", hold)):
                e = (candidate - target)[mask]
                sums[name + "_arm"] += float(e[:, arms].square().sum())
                sums[name + "_gripper"] += float(e[:, [7, 15]].abs().sum())
            if model.kinematics is not None:
                position, rotation = model.kinematics(prediction)
                target_position, target_rotation = model.kinematics(target, clamp=True)
                distance2 = (position - target_position).square().sum(-1)
                angle = torch.acos(((rotation * target_rotation).sum((-1, -2)) / 2 - .5).clamp(-1, 1))
                valid_arms = mask[..., None].expand_as(distance2)
                tcp_squares += float(distance2[valid_arms].sum())
                tcp_angle_squares += float(angle[valid_arms].square().sum())
                tcp_count += int(valid_arms.sum())
                approach = batch["action_approach"][:, :model.n_action_steps] & valid_arms
                if approach.any():
                    approach_squares += float(distance2[approach].sum())
                    approach_angle_squares += float(angle[approach].square().sum())
                    approach_grip_sum += float((prediction - target)[..., [7, 15]].abs()[approach].sum())
                    approach_count += int(approach.sum())
                    approach_distances.append(distance2[approach].sqrt().cpu())
            sums["loss"] += float(loss) * len(target)
            count += len(target)
            frames += int(mask.sum())
    if not count or not frames:
        raise ValueError("Empty validation split")
    metrics = {"val_loss": sums["loss"] / count, "val_normalized_mse": sums["normalized_mse"] / (frames * 16),
               "val_chunks": count, "val_valid_frames": frames}
    for name in ("policy", "hold"):
        metrics[name + "_arm_rmse_rad"] = math.sqrt(sums[name + "_arm"] / (frames * 14))
        metrics[name + "_gripper_mae_mm"] = sums[name + "_gripper"] / (frames * 2) * 1000
    if tcp_count:
        metrics.update(val_tcp_position_rmse_mm=math.sqrt(tcp_squares / tcp_count) * 1000,
                       val_tcp_orientation_rmse_deg=math.degrees(math.sqrt(tcp_angle_squares / tcp_count)),
                       val_approach_samples=approach_count,
                       val_approach_tcp_rmse_mm=math.sqrt(approach_squares / approach_count) * 1000 if approach_count else None,
                       val_approach_orientation_rmse_deg=math.degrees(math.sqrt(approach_angle_squares / approach_count)) if approach_count else None,
                       val_approach_gripper_mae_mm=approach_grip_sum / approach_count * 1000 if approach_count else None,
                       val_approach_tcp_p95_mm=float(torch.quantile(torch.cat(approach_distances), .95)) * 1000 if approach_count else None)
        position_rmse = math.sqrt(approach_squares / approach_count) if approach_count else math.sqrt(tcp_squares / tcp_count)
        angle_rmse = math.sqrt(approach_angle_squares / approach_count) if approach_count else math.sqrt(tcp_angle_squares / tcp_count)
        grip_mm = approach_grip_sum / approach_count * 1000 if approach_count else metrics["policy_gripper_mae_mm"]
        metrics["val_selection_uses_approach"] = bool(approach_count)
        metrics["val_selection_score"] = (position_rmse / model.tcp_position_scale_m
                                           + .1 * angle_rmse / model.tcp_orientation_scale_rad + .25 * grip_mm / 5)
    if not all(x is None or math.isfinite(x) for x in metrics.values()):
        raise ValueError("Nonfinite validation metrics")
    return metrics


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=root / "conf/meanflow_v2.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra-dataset", type=Path, action="append", default=[],
                    help="Extra TRAINING data (converted with --all-train, e.g. simulated corrective demos); mixed into "
                         "each batch at train.extra_fraction. Validation and normalisation use the main dataset only.")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tr, mc = cfg["train"], cfg["model"]
    if args.steps is not None:
        tr["steps"] = args.steps
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if min(tr[k] for k in ("steps", "batch_size", "save_every", "val_samples_per_episode")) < 1:
        ap.error("Steps, batch size and validation/checkpoint intervals must be positive")
    if tr["num_workers"] < 0 or tr["warmup_steps"] < 0 or min(tr["lr"], tr["max_grad_norm"]) <= 0:
        ap.error("Invalid optimizer/dataloader settings")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output exists and is not empty; choose a new run or --resume")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    train = MeanFlowDataset(args.dataset, mc["horizon"], mc["n_obs_steps"])
    val = MeanFlowDataset(args.dataset, mc["horizon"], mc["n_obs_steps"], "val")
    if not len(train) or not len(val):
        raise ValueError("Both train and validation episodes are required")
    extras = [MeanFlowDataset(p, mc["horizon"], mc["n_obs_steps"]) for p in args.extra_dataset]
    for path, extra in zip(args.extra_dataset, extras):
        same = ("fps", "depth", "rgb", "joint_names", "action_representation")
        if (extra.metadata.get("objects") or {}).get("config") != (train.metadata.get("objects") or {}).get("config"):
            raise ValueError(f"{path}: object feature config differs from the main dataset")
        if any(extra.metadata.get(k) != train.metadata.get(k) for k in same):
            raise ValueError(f"{path}: preprocessing differs from the main dataset ({', '.join(same)} must match)")
    extra_metadata = [{"path": str(p), "episodes": len(e.metadata["episodes"]), "frames": len(e)}
                      for p, e in zip(args.extra_dataset, extras)]
    if bool(mc.get("object_features")) != train.has_objects:
        raise ValueError(f"model.object_features={mc.get('object_features', False)} but dataset has_objects="
                         f"{train.has_objects}; convert with --object-features (or set object_features accordingly)")
    if bool(mc.get("use_rgb")) != train.has_rgb:
        raise ValueError(f"model.use_rgb={mc.get('use_rgb', False)} but dataset has_rgb={train.has_rgb}; "
                         "convert with --rgb (or set use_rgb accordingly)")
    model = MeanFlowPolicy(**mc)
    start, best, state = 0, math.inf, None
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
        if state["model_config"] != mc or state["metadata"] != train.metadata \
                or [e["path"] for e in state.get("extra_datasets", [])] != [e["path"] for e in extra_metadata]:
            raise ValueError("Resume model/dataset differs from checkpoint")
        saved_train = state["train_config"]
        if any(tr[k] != saved_train[k] for k in tr if k != "steps") or state["seed"] != cfg["seed"]:
            raise ValueError("Resume training settings differ; only total steps may change")
        start, best = state["step"], state["best_score"]
        if tr["steps"] <= start:
            raise ValueError("Requested steps must exceed the saved step")
        model.load_state_dict(state["model"])
    else:
        model.set_statistics(train.statistics(mc.get("delta_actions", True)))
    model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    shuffle_rng = torch.Generator().manual_seed(cfg["seed"])
    if state:
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        shuffle_rng.set_state(state["shuffle_rng"])
        if device.startswith("cuda") and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if extras:
        fraction = tr.get("extra_fraction", 0.3)
        if not 0 < fraction < 1:
            raise ValueError("train.extra_fraction must be in (0, 1)")
        n_extra = sum(len(e) for e in extras)
        weights = torch.cat([torch.full((len(train),), (1 - fraction) / len(train)),
                             torch.full((n_extra,), fraction / n_extra)])
        sampler = WeightedRandomSampler(weights, num_samples=len(train) + n_extra, replacement=True,
                                        generator=shuffle_rng)
        loader = DataLoader(ConcatDataset([train, *extras]), batch_size=tr["batch_size"], sampler=sampler,
                            num_workers=tr["num_workers"], pin_memory=device.startswith("cuda"))
        print(f"Extra training data: {extra_metadata}; {fraction:.0%} of each batch", flush=True)
    else:
        loader = DataLoader(train, batch_size=tr["batch_size"], shuffle=True, generator=shuffle_rng,
                            num_workers=tr["num_workers"], pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(Subset(val, val.validation_indices(tr["val_samples_per_episode"])),
                            batch_size=tr["batch_size"], shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train_config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"MeanFlow on {device}: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"train={len(train)} frames/{len(train.ranges)} episodes; val={len(val.ranges)} episodes; step={start}", flush=True)
    iterator = iter(loader)
    for step in range(start + 1, tr["steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        fraction = min(1, max(0, (step - tr["warmup_steps"]) / max(1, tr["steps"] - tr["warmup_steps"])))
        factor = min(1, step / max(1, tr["warmup_steps"])) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for group in optimizer.param_groups:
            group["lr"] = tr["lr"] * factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.compute_loss(to_device(batch, device))
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        decay = min(0.9999, 1 - (1 + max(0, step - 1)) ** -0.75)
        with torch.no_grad():
            for p, e in zip(model.parameters(), ema.parameters(), strict=True):
                e.lerp_(p, 1 - decay)
            for b, e in zip(model.buffers(), ema.buffers(), strict=True):
                e.copy_(b)
        if step == start + 1 or step % 100 == 0:
            record = {"step": step, "train_loss": float(loss.detach()), "lr": tr["lr"] * factor,
                      "grad_norm": float(norm), **parts}
            with (args.output / "train_loss.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % tr["save_every"] == 0 or step == tr["steps"]:
            metrics = {"step": step, "train_loss": float(loss.detach()), **validate(ema, val_loader, device, cfg["seed"])}
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            save_meanflow_release(args.output / "release", ema, mc, train.metadata, step)
            selection_metric = "val_selection_score" if "val_selection_score" in metrics else "val_normalized_mse"
            if metrics[selection_metric] < best:
                best = metrics[selection_metric]
                save_meanflow_release(args.output / "best_release", ema, mc, train.metadata, step)
                (args.output / "best_metrics.json").write_text(json.dumps(metrics, indent=2))
            checkpoint = {"step": step, "model_config": mc, "train_config": tr, "seed": cfg["seed"],
                          "metadata": train.metadata, "extra_datasets": extra_metadata,
                          "model": model.state_dict(), "ema": ema.state_dict(),
                          "optimizer": optimizer.state_dict(), "best_score": best,
                          "selection_metric": selection_metric, "rng": torch.get_rng_state(),
                          "shuffle_rng": shuffle_rng.get_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
            torch.save(checkpoint, args.output / "last.pt.tmp")
            (args.output / "last.pt.tmp").replace(args.output / "last.pt")
            try:
                plot_losses(args.output, title="MeanFlow train / validation loss")
                if model.separate_gripper:
                    plot_meanflow_metrics(args.output)
            except ImportError:
                print("Plot skipped: install matplotlib; JSONL metrics and checkpoints are saved", flush=True)
    print(f"Best EMA release: {args.output / 'best_release'}", flush=True)


if __name__ == "__main__":
    main()

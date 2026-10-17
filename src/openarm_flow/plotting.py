"""Render persisted FlowPolicy loss metrics without a display server."""
import json
from pathlib import Path


def plot_losses(directory, title="FlowPolicy train / validation loss"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    train, val = {}, {}
    # New runs record train loss every 100 steps. Older runs only have metrics.
    for filename in ("metrics.jsonl", "train_loss.jsonl"):
        path = directory / filename
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live reader may encounter an unfinished last line.
            step = row["step"]
            if "train_loss" in row:
                train[step] = row["train_loss"]
            if "val_loss" in row:
                val[step] = row["val_loss"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in (("Train (batch)", train), ("Validation (EMA)", val)):
        if values:
            steps = sorted(values)
            ax.plot(steps, [values[s] for s in steps], label=label,
                    marker="o" if label.startswith("Validation") else None)
    ax.set(xlabel="Step", ylabel="Loss", title=title)
    if train or val:
        if all(v > 0 for v in [*train.values(), *val.values()]):
            ax.set_yscale("log")
        ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    output = directory / "loss_curve.png"
    temporary = directory / "loss_curve.tmp.png"
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(output)
    return output


def plot_meanflow_metrics(directory):
    """Physical-unit validation diagnostics, including the actual selection score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    rows = []
    for line in (directory / "metrics.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("TCP FK error (mm)", ["val_tcp_position_rmse_mm", "val_approach_tcp_rmse_mm", "val_approach_tcp_p95_mm"]),
        ("Gripper error (mm)", ["policy_gripper_mae_mm", "val_approach_gripper_mae_mm"]),
        ("TCP orientation (degrees)", ["val_tcp_orientation_rmse_deg", "val_approach_orientation_rmse_deg"]),
        ("Checkpoint selection score (lower is better)", ["val_selection_score"]),
    ]
    for axis, (title, keys) in zip(axes.flat, panels, strict=True):
        for key in keys:
            values = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
            if values:
                axis.plot([p[0] for p in values], [p[1] for p in values], marker="o", label=key.removeprefix("val_"))
        axis.set(title=title, xlabel="Step")
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.suptitle("MeanFlow v2 — validation on held-out episodes; TCP metrics are URDF FK proxies")
    fig.tight_layout()
    output = directory / "action_metrics.png"
    temporary = directory / "action_metrics.tmp.png"
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    temporary.replace(output)
    return output

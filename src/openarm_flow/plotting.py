"""Render persisted FlowPolicy loss metrics without a display server."""
import json
from pathlib import Path


def plot_losses(directory):
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
    ax.set(xlabel="Step", ylabel="Loss", title="FlowPolicy train / validation loss")
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

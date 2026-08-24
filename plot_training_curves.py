#!/usr/bin/env python3
"""Plot train/validation loss and mIoU from the rank-0 training log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_RE = re.compile(
    r"\[epoch:(?P<epoch>\d+),\s*iter:\d+\]\s*"
    r"Loss:\s*(?P<loss>[0-9.]+)\s*\|\s*"
    r"mIoU:\s*(?P<miou>[0-9.]+)%"
)


def parse_log(log_path: Path) -> list[dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    split = "train"

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Waiting Val..." in line:
            split = "val"
            continue
        if line.startswith("Epoch:"):
            split = "train"

        match = METRIC_RE.search(line)
        if not match:
            continue

        epoch = int(match.group("epoch"))
        records.setdefault(epoch, {})
        records[epoch][f"{split}_loss"] = float(match.group("loss"))
        records[epoch][f"{split}_miou"] = float(match.group("miou"))

    missing = [
        epoch
        for epoch, record in records.items()
        if not {"train_loss", "train_miou", "val_loss", "val_miou"}.issubset(record)
    ]
    if missing:
        raise ValueError(f"Missing train/validation metrics for epoch(s): {missing}")
    if not records:
        raise ValueError(f"No epoch metrics found in {log_path}")

    return [{"epoch": epoch, **records[epoch]} for epoch in sorted(records)]


def write_csv(records: list[dict[str, float]], csv_path: Path) -> None:
    fields = ["epoch", "train_loss", "val_loss", "train_miou", "val_miou"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_records(records: list[dict[str, float]], output_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in records]
    train_loss = [row["train_loss"] for row in records]
    val_loss = [row["val_loss"] for row in records]
    train_miou = [row["train_miou"] for row in records]
    val_miou = [row["val_miou"] for row in records]

    best_index = max(range(len(records)), key=lambda index: val_miou[index])
    best_epoch = epochs[best_index]
    best_miou = val_miou[best_index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Validation", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_miou, label="Train", linewidth=2)
    axes[1].plot(epochs, val_miou, label="Validation", linewidth=2)
    axes[1].scatter([best_epoch], [best_miou], zorder=3, label=f"Best val: {best_miou:.2f}% (epoch {best_epoch})")
    axes[1].set_title("mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="rank-0 stdout.log path")
    parser.add_argument("-o", "--output", type=Path, default=Path("training_curves.png"))
    parser.add_argument("--csv", type=Path, default=None, help="optional parsed metrics CSV path")
    args = parser.parse_args()

    records = parse_log(args.log)
    plot_records(records, args.output, f"Training curves ({len(records)} epochs)")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(records, args.csv)

    best = max(records, key=lambda row: row["val_miou"])
    print(f"Parsed {len(records)} epochs")
    print(f"Best validation mIoU: {best['val_miou']:.3f}% at epoch {int(best['epoch'])}")
    print(f"Saved plot: {args.output}")
    if args.csv is not None:
        print(f"Saved data: {args.csv}")


if __name__ == "__main__":
    main()

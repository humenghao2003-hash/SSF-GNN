"""Generate figures used by the SSF-GNN experiment record."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_figures"
CLASSES = ["Normal", "Wildfire", "Flood", "Oilspill", "Redtide", "Volcaniceruption", "Algalbloom"]
TRAIN = np.array([9407, 1104, 2143, 1482, 4126, 365, 2243])
TEST = np.array([11573, 754, 753, 1354, 3785, 162, 2441])
CM = np.array([
    [10914, 13, 65, 147, 119, 87, 228],
    [79, 671, 0, 3, 0, 1, 0],
    [20, 0, 730, 0, 3, 0, 0],
    [636, 0, 3, 682, 20, 0, 13],
    [134, 0, 0, 2, 3503, 0, 146],
    [26, 23, 0, 0, 0, 113, 0],
    [133, 2, 74, 1, 114, 0, 2117],
])
PRECISION = np.array([.91391727, .94640339, .83715596, .81676647, .93189678, .56218905, .84544728])
RECALL = np.array([.94305712, .88992042, .96945551, .50369276, .92549538, .69753086, .86726751])
F1 = np.array([.92825856, .91729323, .89846154, .62311558, .92868505, .62258953, .85621840])


def style(ax):
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)


def read_curves():
    with (ROOT / "training_curves.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def distribution():
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    first = ax.bar(x - .18, TRAIN, .36, label="Train", color="#2563eb")
    second = ax.bar(x + .18, TEST, .36, label="Test", color="#f97316")
    ax.set_title("S2MHD class distribution")
    ax.set_ylabel("Number of TIFF images")
    ax.set_xticks(x, CLASSES, rotation=25, ha="right")
    ax.legend(frameon=False, ncols=2)
    style(ax)
    for group in (first, second):
        ax.bar_label(group, padding=2, fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_distribution.png", bbox_inches="tight")
    plt.close(fig)


def curves():
    data = read_curves()
    epoch = np.array([int(row["epoch"]) for row in data])
    train_loss = np.array([float(row["train_loss"]) for row in data])
    val_loss = np.array([float(row["val_loss"]) for row in data])
    train_miou = np.array([float(row["train_miou"]) for row in data])
    val_miou = np.array([float(row["val_miou"]) for row in data])
    best = val_miou.argmax()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    axes[0].plot(epoch, train_loss, label="Train", linewidth=2, color="#2563eb")
    axes[0].plot(epoch, val_loss, label="Test/validation", linewidth=2, color="#f97316")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    style(axes[0])
    axes[1].plot(epoch, train_miou, label="Train", linewidth=2, color="#2563eb")
    axes[1].plot(epoch, val_miou, label="Test/validation", linewidth=2, color="#f97316")
    axes[1].scatter([epoch[best]], [val_miou[best]], color="#dc2626", zorder=3)
    axes[1].annotate(f"best={val_miou[best]:.3f}%\nepoch {epoch[best]}", (epoch[best], val_miou[best]),
                     xytext=(-65, -40), textcoords="offset points", fontsize=8,
                     arrowprops={"arrowstyle": "->", "color": "#dc2626"})
    axes[1].set(title="Mean IoU", xlabel="Epoch", ylabel="mIoU (%)")
    axes[1].legend(frameon=False)
    style(axes[1])
    fig.suptitle("SSF-GNN training history", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def confusion():
    normalized = CM / CM.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=180)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Row-normalized proportion")
    ax.set(title="Test-set confusion matrix", xlabel="Predicted label", ylabel="True label")
    ax.set_xticks(np.arange(7), CLASSES, rotation=35, ha="right")
    ax.set_yticks(np.arange(7), CLASSES)
    for i in range(7):
        for j in range(7):
            color = "white" if normalized[i, j] > .5 else "#111827"
            ax.text(j, i, f"{CM[i, j]}\n({normalized[i, j] * 100:.1f}%)",
                    ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def metrics():
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.bar(x - .25, PRECISION * 100, .25, label="Precision", color="#2563eb")
    ax.bar(x, RECALL * 100, .25, label="Recall", color="#16a34a")
    ax.bar(x + .25, F1 * 100, .25, label="F1", color="#f97316")
    ax.set(title="Per-class test metrics", ylabel="Score (%)", ylim=(0, 105))
    ax.set_xticks(x, CLASSES, rotation=25, ha="right")
    ax.legend(frameon=False, ncols=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "test_metrics.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    distribution()
    curves()
    confusion()
    metrics()
    print(f"Generated figures in {OUT}")

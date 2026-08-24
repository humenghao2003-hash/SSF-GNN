"""Generate image-level inference visualizations for the SSF-GNN classifier.

The model in this repository performs image-level seven-class classification,
not bounding-box detection.  This script therefore produces Sentinel-2
true-colour thumbnails annotated with the ground-truth label, predicted label,
confidence, and whether the prediction is correct.  It also writes a compact
CSV manifest so each tile can be traced back to its source GeoTIFF.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from osgeo import gdal

from Compute_indices import compute_indices
from model import SSFGNN


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "test"
OUT = ROOT / "experiment_figures" / "inference"
CLASSES = [
    "Normal",
    "Wildfire",
    "Flood",
    "Oilspill",
    "Redtide",
    "Volcaniceruption",
    "Algalbloom",
]

MEAN = np.array(
    [0.0736, 0.0802, 0.0935, 0.0933, 0.1120, 0.1290, 0.1386, 0.1369,
     0.1398, 0.1532, 0.1195, 0.0972], dtype=np.float32
)
STD = np.array(
    [0.1041, 0.1055, 0.1128, 0.1293, 0.1377, 0.1508, 0.1594, 0.1627,
     0.1651, 0.1768, 0.1612, 0.1433], dtype=np.float32
)


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**12, path.name)


def read_raw(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"GDAL cannot open {path}")
    image = dataset.ReadAsArray().astype(np.float32)
    image = np.clip((image - 1000.0) / 10000.0, -0.1, 1.0)
    return image


def make_model_input(raw: np.ndarray) -> np.ndarray:
    indices = compute_indices(raw)
    normalized = (raw - MEAN[:, None, None]) / STD[:, None, None]
    return np.concatenate([normalized, indices], axis=0).astype(np.float32)


def true_colour(raw: np.ndarray) -> np.ndarray:
    """Create a robust Sentinel-2 B4/B3/B2 display image."""
    # Dataset band order is B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.
    rgb = raw[[3, 2, 1]].copy()
    out = np.empty_like(rgb)
    for channel in range(3):
        low, high = np.percentile(rgb[channel], [2, 98])
        out[channel] = np.clip((rgb[channel] - low) / max(high - low, 1e-6), 0, 1)
    # Mild gamma correction makes vegetation and water easier to inspect.
    return np.power(out.transpose(1, 2, 0), 0.8)


def choose_samples(per_class: int = 3) -> list[tuple[Path, int]]:
    selected: list[tuple[Path, int]] = []
    for label, class_name in enumerate(CLASSES):
        files = sorted((DATA_ROOT / class_name).glob("*.tif"), key=numeric_key)
        if not files:
            continue
        count = min(per_class, len(files))
        indices = np.linspace(0, len(files) - 1, count, dtype=int)
        selected.extend((files[index], label) for index in indices)
    return selected


def infer(samples: list[tuple[Path, int]], device: torch.device) -> list[dict]:
    model = SSFGNN(len(CLASSES))
    checkpoint = torch.load(ROOT / "checkpoints_new" / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    results: list[dict] = []
    with torch.inference_mode():
        for path, label in samples:
            raw = read_raw(path)
            model_input = torch.from_numpy(make_model_input(raw))[None].to(device)
            logits = model(model_input)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            prediction = int(probabilities.argmax())
            results.append({
                "path": path,
                "raw": raw,
                "label": label,
                "prediction": prediction,
                "confidence": float(probabilities[prediction]),
                "probabilities": probabilities,
            })
    return results


def add_label(ax, result: dict) -> None:
    correct = result["label"] == result["prediction"]
    colour = "#15803d" if correct else "#dc2626"
    ax.set_title(
        f"GT: {CLASSES[result['label']]}\n"
        f"Pred: {CLASSES[result['prediction']]} ({result['confidence']:.1%})",
        fontsize=9,
        color=colour,
        pad=5,
    )
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           fill=False, linewidth=3, edgecolor=colour,
                           clip_on=False))
    ax.axis("off")


def plot_grid(results: list[dict], output: Path) -> None:
    columns = 3
    rows = int(np.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 4.15), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for axis, result in zip(axes, results):
        axis.imshow(true_colour(result["raw"]))
        add_label(axis, result)
    for axis in axes[len(results):]:
        axis.axis("off")
    fig.suptitle(
        "SSF-GNN image-level inference | Sentinel-2 true-colour samples\n"
        "Green border = correct, red border = misclassified",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors(results: list[dict], output: Path) -> None:
    errors = [result for result in results if result["label"] != result["prediction"]]
    if not errors:
        errors = sorted(results, key=lambda result: result["confidence"])[:min(6, len(results))]
    columns = 3
    rows = max(1, int(np.ceil(len(errors) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 4.2), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    for axis, result in zip(axes, errors):
        axis.imshow(true_colour(result["raw"]))
        add_label(axis, result)
    for axis in axes[len(errors):]:
        axis.axis("off")
    fig.suptitle(
        "SSF-GNN inference examples requiring review\n"
        "Red border indicates the predicted class differs from the ground truth",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confidence(results: list[dict], output: Path) -> None:
    names = [f"{CLASSES[r['label']]}\n{r['path'].stem}" for r in results]
    values = [r["confidence"] * 100 for r in results]
    colours = ["#16a34a" if r["label"] == r["prediction"] else "#dc2626" for r in results]
    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    positions = np.arange(len(results))
    ax.bar(positions, values, color=colours, alpha=0.9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Prediction confidence (%)")
    ax.set_xlabel("Selected test images (class / file stem)")
    ax.set_title("SSF-GNN confidence on visualized test samples")
    ax.set_xticks(positions, names, rotation=65, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_manifest(results: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "ground_truth", "prediction", "confidence", "correct"])
        for result in results:
            writer.writerow([
                str(result["path"].relative_to(ROOT)),
                CLASSES[result["label"]],
                CLASSES[result["prediction"]],
                f"{result['confidence']:.6f}",
                result["label"] == result["prediction"],
            ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = choose_samples(per_class=3)
    if not samples:
        raise RuntimeError(f"No .tif files found below {DATA_ROOT}")
    print(f"Running inference for {len(samples)} selected images on {device}...")
    results = infer(samples, device)
    plot_grid(results, OUT / "inference_samples.png")
    plot_errors(results, OUT / "inference_errors.png")
    plot_confidence(results, OUT / "inference_confidence.png")
    write_manifest(results, OUT / "inference_samples.csv")
    correct = sum(result["label"] == result["prediction"] for result in results)
    print(f"Selected accuracy: {correct}/{len(results)} = {correct / len(results):.2%}")
    print(f"Saved visualizations to {OUT}")


if __name__ == "__main__":
    main()

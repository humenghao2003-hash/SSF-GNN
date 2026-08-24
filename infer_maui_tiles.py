"""Run SSF-GNN on a 6x6 Sentinel-2 Maui scene and render a tile map."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from osgeo import gdal

from Compute_indices import compute_indices
from dataset import class_dict
from model import SSFGNN


SOURCE = Path("data/exp/maui_wildfire_lahaina_2023-08-18_S2L2A_12band_15km.tif")
OUT = Path("data/exp/maui_wildfire_inference")
TILE = 256
MEAN = np.array([0.0736, 0.0802, 0.0935, 0.0933, 0.1120, 0.1290, 0.1386, 0.1369, 0.1398, 0.1532, 0.1195, 0.0972], dtype=np.float32)
STD = np.array([0.1041, 0.1055, 0.1128, 0.1293, 0.1377, 0.1508, 0.1594, 0.1627, 0.1651, 0.1768, 0.1612, 0.1433], dtype=np.float32)
COLORS = ["#64748b", "#ef4444", "#2563eb", "#7c3aed", "#db2777", "#f97316", "#16a34a"]


def make_input(raw: np.ndarray) -> np.ndarray:
    reflectance = np.clip((raw.astype(np.float32) - 1000) / 10000, -0.1, 1.0)
    indices = compute_indices(reflectance)
    normalized = (reflectance - MEAN[:, None, None]) / STD[:, None, None]
    return np.concatenate((normalized, indices), axis=0)


def rgb(raw: np.ndarray) -> np.ndarray:
    # Sentinel-2 input order: B2 (blue), B3 (green), B4 (red).
    image = np.moveaxis(raw[[3, 2, 1]], 0, -1).astype(np.float32)
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image - lo) / max(hi - lo, 1), 0, 1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = gdal.Open(str(SOURCE))
    raw = ds.ReadAsArray().astype(np.float32)
    rows, cols = raw.shape[1] // TILE, raw.shape[2] // TILE
    tiles, locations = [], []
    for row in range(rows):
        for col in range(cols):
            tile = raw[:, row*TILE:(row+1)*TILE, col*TILE:(col+1)*TILE]
            tiles.append(make_input(tile))
            locations.append((row, col))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSFGNN(7)
    model.load_state_dict(torch.load("checkpoints_new/best.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    inputs = torch.from_numpy(np.stack(tiles))
    probabilities = []
    with torch.inference_mode():
        for batch in inputs.split(6):
            probabilities.append(torch.softmax(model(batch.to(device)), dim=1).cpu())
    probabilities = torch.cat(probabilities).numpy()
    predicted, confidence = probabilities.argmax(axis=1), probabilities.max(axis=1)

    with (OUT / "tile_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "column", "prediction", "confidence", "probabilities"])
        for (row, col), label, score, probs in zip(locations, predicted, confidence, probabilities):
            writer.writerow([row, col, class_dict[label], f"{score:.6f}", ";".join(f"{v:.5f}" for v in probs)])

    background = rgb(raw)
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(background)
    for (row, col), label, score in zip(locations, predicted, confidence):
        y, x = row*TILE, col*TILE
        rect = plt.Rectangle((x, y), TILE, TILE, facecolor=COLORS[label], edgecolor="white", linewidth=.9, alpha=.42)
        ax.add_patch(rect)
        ax.text(x + TILE/2, y + TILE/2, f"{class_dict[label]}\n{score:.0%}", ha="center", va="center", fontsize=7, color="white", weight="bold", bbox={"facecolor":"black", "alpha":.45, "pad":1.5, "edgecolor":"none"})
    ax.set_title("SSF-GNN tile-level inference | Lahaina, Maui wildfire scene | Sentinel-2 L2A, 2023-08-18")
    ax.set_axis_off()
    legend = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=.7, label=name) for name, color in zip(class_dict, COLORS)]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=.88, ncol=2)
    fig.savefig(OUT / "tile_localization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Counts:")
    for index, name in enumerate(class_dict):
        print(f"{name}: {(predicted == index).sum()}")
    print(f"Wrote {OUT / 'tile_localization.png'}")


if __name__ == "__main__":
    main()

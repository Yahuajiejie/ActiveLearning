#!/usr/bin/env python3
"""Draw a TensorBoard-style scalar curve with matplotlib.

Usage:
    conda run -n affinity-registry python scripts/plot_masked_accuracy_matplotlib.py \
        /Users/yahuagege/Downloads/20260112-182553.csv \
        figures/masked_accuracy_curve_tensorboard_style
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


def smooth(values: list[float], weight: float = 0.6) -> list[float]:
    """TensorBoard-like exponential smoothing.

    weight=0 shows the raw curve; larger values produce a smoother curve.
    TensorBoard's scalar smoothing is also an exponential moving average style
    visual aid, so this makes the exported figure feel closer to the UI.
    """
    if not values:
        return []
    smoothed = []
    last = values[0]
    for value in values:
        last = last * weight + (1.0 - weight) * value
        smoothed.append(last)
    return smoothed


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/yahuagege/Downloads/20260112-182553.csv")
    out_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/masked_accuracy_curve_tensorboard_style")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["Step"] = df["Step"].astype(int)
    df["Value"] = df["Value"].astype(float)

    first = df.iloc[0]
    last = df.iloc[-1]
    best = df.loc[df["Value"].idxmax()]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = df["Step"]
    y = df["Value"]
    y_smooth = smooth(y.tolist(), weight=0.55)

    # TensorBoard-like look: a faint raw curve plus a clean smoothed curve.
    # No area fill, because the y-value itself is what matters here.
    ax.plot(x, y, color="#8EC9E8", linewidth=2.0, alpha=0.45, zorder=2, label="raw")
    ax.plot(x, y_smooth, color="#2E9CCA", linewidth=3.0, solid_capstyle="round", zorder=3, label="smoothed")

    ax.scatter([last["Step"]], [last["Value"]], s=52, color="#2E9CCA", edgecolor="white", linewidth=1.2, zorder=4)
    ax.text(
        last["Step"] + 0.15,
        last["Value"],
        f"{last['Value']:.1f}%",
        va="center",
        ha="left",
        fontsize=10.5,
        color="#2E9CCA",
        weight="bold",
    )

    fig.suptitle("Masked Token Prediction Accuracy", fontsize=18, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.925,
        "ESMC-600M MLM fine-tuning on fluorescent protein sequences",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#555555",
    )

    ax.set_xlabel("Epoch", fontsize=13, weight="bold", labelpad=12)
    ax.set_ylabel("Masked accuracy (%)", fontsize=13, weight="bold", labelpad=12)
    ax.set_xlim(1, 20.2)
    ax.set_ylim(74, 95.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([75, 80, 85, 90, 95])
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=1.0)
    ax.grid(False, axis="x")
    ax.tick_params(colors="#555555")
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    note = (
        f"Masked accuracy: {first['Value']:.1f}% -> {last['Value']:.1f}% "
        f"(peak {best['Value']:.1f}% at epoch {int(best['Step'])})."
    )
    fig.text(0.125, 0.02, note, ha="left", va="bottom", fontsize=10.5, color="#555555")

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.78)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    print(f"saved {out_prefix.with_suffix('.png')}")
    print(f"saved {out_prefix.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

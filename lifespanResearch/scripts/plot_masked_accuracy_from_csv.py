#!/usr/bin/env python3
"""Plot a TensorBoard scalar CSV as a clean SVG figure.

This script intentionally uses only the Python standard library so it can run
on lightweight login nodes where matplotlib/pandas may not be installed.
"""

from __future__ import annotations

import csv
import html
import sys
from pathlib import Path


def nice_ticks(start: float, stop: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/yahuagege/Downloads/20260112-182553.csv")
    out_svg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/masked_accuracy_curve.svg")

    rows: list[tuple[int, float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(float(row["Step"])), float(row["Value"])))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    steps = [s for s, _ in rows]
    values = [v for _, v in rows]
    first_step, first_val = rows[0]
    last_step, last_val = rows[-1]
    best_step, best_val = max(rows, key=lambda item: item[1])

    width, height = 1280, 760
    left, right, top, bottom = 118, 72, 92, 132
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = min(steps), max(steps)
    y_min = max(0.0, min(values) - 3.0)
    y_max = min(100.0, max(values) + 3.0)

    def sx(step: float) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    line_points = [(sx(s), sy(v)) for s, v in rows]
    area_points = [(sx(x_min), sy(y_min))] + line_points + [(sx(x_max), sy(y_min))]

    x_ticks = [1, 5, 10, 15, 20]
    y_ticks = nice_ticks(75, 95, 5)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="area" x1="0" x2="0" y1="0" y2="1">',
        '<stop offset="0%" stop-color="#2E9CCA" stop-opacity="0.22"/>',
        '<stop offset="100%" stop-color="#2E9CCA" stop-opacity="0.02"/>',
        "</linearGradient>",
        '<filter id="shadow" x="-15%" y="-15%" width="130%" height="130%">',
        '<feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#000000" flood-opacity="0.10"/>',
        "</filter>",
        "<style>",
        ".font{font-family:Arial,Helvetica,sans-serif;fill:#111}",
        ".title{font-size:34px;font-weight:800}",
        ".subtitle{font-size:18px;fill:#555}",
        ".axis{stroke:#111;stroke-width:2.2}",
        ".grid{stroke:#d9dee3;stroke-width:1.4}",
        ".tick{font-size:15px;fill:#4a4a4a}",
        ".label{font-size:18px;font-weight:700;fill:#222}",
        ".note{font-size:16px;fill:#555}",
        "</style>",
        "</defs>",
        '<rect width="1280" height="760" fill="#ffffff"/>',
        '<rect x="42" y="38" width="1196" height="664" rx="28" fill="#fbfdff" filter="url(#shadow)"/>',
        '<text class="font title" x="640" y="74" text-anchor="middle">Masked Token Prediction Accuracy</text>',
        '<text class="font subtitle" x="640" y="104" text-anchor="middle">ESMC-600M MLM fine-tuning on fluorescent protein sequences</text>',
    ]

    for tick in y_ticks:
        y = sy(tick)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        svg.append(f'<text class="font tick" x="{left-15}" y="{y+5:.2f}" text-anchor="end">{tick:.0f}</text>')

    for tick in x_ticks:
        x = sx(tick)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        svg.append(f'<text class="font tick" x="{x:.2f}" y="{height-bottom+28}" text-anchor="middle">{tick}</text>')

    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<polygon points="{polyline(area_points)}" fill="url(#area)"/>',
            f'<polyline points="{polyline(line_points)}" fill="none" stroke="#2E9CCA" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>',
        ]
    )

    for step, val in rows:
        svg.append(f'<circle cx="{sx(step):.2f}" cy="{sy(val):.2f}" r="4.8" fill="#2E9CCA" stroke="#ffffff" stroke-width="2"/>')

    # Highlight first, final, and best points.
    highlights = [
        (first_step, first_val, "start", "#7c8a95", -24, 30),
        (last_step, last_val, "final", "#1f7a4d", -70, -28),
        (best_step, best_val, "best", "#ef8a4c", -52, -34),
    ]
    for step, val, label, color, dx, dy in highlights:
        x, y = sx(step), sy(val)
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8.5" fill="{color}" stroke="#ffffff" stroke-width="3"/>')
        text = f"{label}: {val:.1f}%"
        tx, ty = x + dx, y + dy
        svg.append(f'<rect x="{tx-12:.2f}" y="{ty-24:.2f}" width="{len(text)*9+24}" height="32" rx="10" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text class="font tick" x="{tx:.2f}" y="{ty-4:.2f}">{html.escape(text)}</text>')

    svg.extend(
        [
            f'<text class="font label" x="{left + plot_w/2:.2f}" y="{height-58}" text-anchor="middle">Epoch</text>',
            f'<text class="font label" x="42" y="{top + plot_h/2:.2f}" text-anchor="middle" transform="rotate(-90 42 {top + plot_h/2:.2f})">Masked accuracy (%)</text>',
            f'<text class="font note" x="{left}" y="{height-25}">Accuracy improved from {first_val:.1f}% to {last_val:.1f}% over 20 epochs; peak accuracy reached {best_val:.1f}% at epoch {best_step}.</text>',
            "</svg>",
        ]
    )

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg), encoding="utf-8")
    print(f"saved {out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

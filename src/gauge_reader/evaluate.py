"""Benchmark the reading pipeline on synthetic gauges with known ground truth.

    python -m gauge_reader.evaluate --n 1000 --out results

Writes results/metrics.json and results/examples.png.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .reader import draw_reading, read_gauge
from .synthetic import render_gauge


def run(n: int, realistic: bool, seed: int):
    rng = np.random.default_rng(seed)
    errors, times, examples, failures = [], [], [], 0
    for _ in range(n):
        img, gt = render_gauge(rng, realistic=realistic)
        t0 = time.perf_counter()
        reading = read_gauge(img, gt.spec)
        times.append(time.perf_counter() - t0)
        if reading is None:
            failures += 1
            continue
        err = abs(reading.value - gt.value) / gt.spec.span * 100
        errors.append(err)
        if len(examples) < 8:
            examples.append((img, reading, gt))
    e = np.array(errors)
    metrics = {
        "images": n,
        "detection_failures": failures,
        "mae_pct_full_scale": round(float(e.mean()), 3),
        "median_pct_full_scale": round(float(np.median(e)), 3),
        "p95_pct_full_scale": round(float(np.percentile(e, 95)), 3),
        "max_pct_full_scale": round(float(e.max()), 3),
        "within_1pct": round(float(np.mean(e < 1) * len(e) / n), 4),
        "within_2pct": round(float(np.mean(e < 2) * len(e) / n), 4),
        "ms_per_image": round(float(np.mean(times) * 1000), 2),
    }
    return metrics, examples


def save_examples(examples, path: Path, tile: int = 300) -> None:
    tiles = []
    for img, reading, gt in examples:
        label = f"read {reading.value:.2f} | true {gt.value:.2f}"
        tiles.append(cv2.resize(draw_reading(img, reading, label), (tile, tile)))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="images per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, realistic in [("clean", False), ("realistic", True)]:
        metrics, examples = run(args.n, realistic, args.seed)
        results[name] = metrics
        print(f"{name:10s} {metrics}")
        if realistic:
            save_examples(examples, out / "examples.png")
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

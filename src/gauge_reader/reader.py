"""Classical gauge-reading pipeline: dial detection -> needle angle -> value.

1. Dial:   Hough circle transform finds the gauge face (centre + radius).
2. Needle: a polar sweep integrates "darkness" along rays from the centre; the needle is
           the darkest ray within the calibrated scale. Rays are sampled only between
           20% and 60% of the radius, away from the tick marks, labels and the needle tail.
3. Refine: a weighted line fit on dark pixels near the coarse angle gives the needle
           direction independently of small errors in the detected centre.
4. Value:  the angle is mapped to a reading with the dial calibration (DialSpec).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .dial import DialSpec, polar_to_xy, xy_to_angle


@dataclass(frozen=True)
class Dial:
    cx: float
    cy: float
    radius: float


@dataclass(frozen=True)
class Reading:
    value: float
    angle_deg: float
    dial: Dial
    confidence: float  # peak-to-background contrast of the needle ray, >1 is better


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def find_dial(img: np.ndarray) -> Dial | None:
    """Locate the gauge face. Returns the strongest circle, or None."""
    gray = cv2.medianBlur(_gray(img), 5)
    h, w = gray.shape
    s = min(h, w)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=s / 4,
        param1=120, param2=40, minRadius=int(0.12 * s), maxRadius=int(0.5 * s),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0, 0]
    return Dial(float(cx), float(cy), float(r))


def refine_center(img: np.ndarray, dial: Dial) -> Dial:
    """Move the dial centre onto the needle pivot (hub).

    The Hough circle follows the rim, which shifts under perspective and blur. The hub is
    the true rotation centre: a compact dark blob near the middle. A morphological opening
    with a disk wider than the needle removes the needle, keeping only the hub.
    """
    dark = _darkness_map(img, dial)
    r = dial.radius
    win = int(0.3 * r)
    x0, y0 = int(max(dial.cx - win, 0)), int(max(dial.cy - win, 0))
    x1, y1 = int(min(dial.cx + win, dark.shape[1])), int(min(dial.cy + win, dark.shape[0]))
    patch = dark[y0:y1, x0:x1]
    if patch.size == 0 or patch.max() <= 0:
        return dial
    mask = (patch > 0.5 * np.percentile(patch, 99.5)).astype(np.uint8)
    k = max(3, int(0.1 * r)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    best, best_d = None, np.inf
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.5 * np.pi * (0.04 * r) ** 2:
            continue
        cx, cy = centroids[i][0] + x0, centroids[i][1] + y0
        d = np.hypot(cx - dial.cx, cy - dial.cy)
        if d < best_d:
            best, best_d = (cx, cy), d
    if best is None or best_d > 0.25 * r:
        return dial
    return Dial(float(best[0]), float(best[1]), dial.radius)


def _darkness_map(img: np.ndarray, dial: Dial) -> np.ndarray:
    """Per-pixel darkness relative to the dial face (robust to global lighting)."""
    gray = cv2.GaussianBlur(_gray(img), (3, 3), 0).astype(np.float32)
    # Red needles are dark in the green channel even when the grey level is moderate.
    if img.ndim == 3:
        gray = np.minimum(gray, img[:, :, 1].astype(np.float32))
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    rr = np.hypot(xx - dial.cx, yy - dial.cy)
    face = gray[(rr > 0.2 * dial.radius) & (rr < 0.6 * dial.radius)]
    face_level = np.median(face) if face.size else gray.max()
    return np.clip(face_level - gray, 0, None)


def _coarse_angle(dark: np.ndarray, dial: Dial, spec: DialSpec, step: float = 0.5):
    margin = 8.0
    angles = (spec.min_angle_deg - margin + np.arange(0, spec.sweep_deg + 2 * margin + step, step)) % 360
    radii = np.linspace(0.2, 0.6, 40) * dial.radius
    a = np.radians(angles)[:, None]
    map_x = (dial.cx + radii[None, :] * np.sin(a)).astype(np.float32)
    map_y = (dial.cy - radii[None, :] * np.cos(a)).astype(np.float32)
    samples = cv2.remap(dark, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    score = samples.mean(axis=1)
    i = int(np.argmax(score))
    background = float(np.median(score)) + 1e-6
    return float(angles[i]), float(score[i] / background)


def _refine_angle(dark: np.ndarray, dial: Dial, coarse: float, wedge: float = 6.0) -> float:
    """Weighted principal direction of dark pixels in a narrow wedge around `coarse`."""
    h, w = dark.shape
    r = dial.radius
    x0, x1 = int(max(dial.cx - r, 0)), int(min(dial.cx + r, w))
    y0, y1 = int(max(dial.cy - r, 0)), int(min(dial.cy + r, h))
    patch = dark[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx, dy = xs - dial.cx, ys - dial.cy
    rr = np.hypot(dx, dy)
    ang = (np.degrees(np.arctan2(dx, -dy)) % 360)
    diff = np.abs((ang - coarse + 180) % 360 - 180)
    sel = (rr > 0.15 * r) & (rr < 0.7 * r) & (diff < wedge)
    wts = patch[sel]
    if wts.size < 20 or wts.sum() <= 0:
        return coarse
    keep = wts > 0.5 * wts.max()
    px, py, wts = xs[sel][keep].astype(np.float64), ys[sel][keep].astype(np.float64), wts[keep].astype(np.float64)
    mx, my = np.average(px, weights=wts), np.average(py, weights=wts)
    cov = np.cov(np.vstack([px - mx, py - my]), aweights=wts)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    refined = xy_to_angle(vx, vy)
    if abs((refined - coarse + 180) % 360 - 180) > 90:  # orient away from the hub
        refined = (refined + 180) % 360
    if abs((refined - coarse + 180) % 360 - 180) > wedge:  # implausible fit, keep coarse
        return coarse
    return refined


def read_gauge(img: np.ndarray, spec: DialSpec, dial: Dial | None = None) -> Reading | None:
    """Read one gauge image. Pass `dial` to skip detection (e.g. from a YOLO box)."""
    dial = dial or find_dial(img)
    if dial is None:
        return None
    dial = refine_center(img, dial)
    dark = _darkness_map(img, dial)
    coarse, confidence = _coarse_angle(dark, dial, spec)
    angle = _refine_angle(dark, dial, coarse)
    return Reading(spec.angle_to_value(angle), angle, dial, confidence)


def draw_reading(img: np.ndarray, reading: Reading, label: str | None = None) -> np.ndarray:
    """Return a copy of the image with the detected dial, needle and value drawn on it."""
    out = img.copy()
    d = reading.dial
    cv2.circle(out, (int(d.cx), int(d.cy)), int(d.radius), (0, 200, 0), 2, cv2.LINE_AA)
    tip = polar_to_xy(d.cx, d.cy, 0.9 * d.radius, reading.angle_deg)
    cv2.line(out, (int(d.cx), int(d.cy)), (int(tip[0]), int(tip[1])), (255, 120, 0), 2, cv2.LINE_AA)
    text = label or f"{reading.value:.2f}"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

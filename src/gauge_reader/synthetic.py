"""Synthetic analog gauge generator with known ground truth.

Used to benchmark the reading pipeline before (and alongside) real labelled photos.
Each image comes with the exact dial centre, radius and needle value.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .dial import DialSpec, polar_to_xy

# Common industrial dial layouts: 270° (most pressure gauges), 180° and 240° scales.
DIAL_TYPES = (
    DialSpec(225, 135, 0, 10),
    DialSpec(225, 135, 0, 16),
    DialSpec(225, 135, 0, 160),
    DialSpec(270, 90, 0, 100),
    DialSpec(240, 120, -1, 5),
)


@dataclass(frozen=True)
class GroundTruth:
    center: tuple[float, float]
    radius: float
    value: float
    spec: DialSpec


def _fmt(v: float) -> str:
    return f"{v:g}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))


def _draw_dial(img, cx, cy, r, spec: DialSpec, value: float, rng: np.random.Generator):
    face = int(rng.integers(215, 256))
    cv2.circle(img, (int(cx), int(cy)), int(r * 1.06), (40, 40, 40), -1, cv2.LINE_AA)  # bezel
    cv2.circle(img, (int(cx), int(cy)), int(r), (face, face, face - int(rng.integers(0, 15))), -1, cv2.LINE_AA)

    # Real scales use "nice" label steps (1, 2, 2.5, 5 x 10^k).
    def nice(step: float) -> bool:
        mant = step / 10 ** np.floor(np.log10(step))
        return any(abs(mant - m) < 1e-9 for m in (1, 2, 2.5, 5))
    options = [n for n in range(4, 11) if nice(spec.span / n)] or [5]
    n_major = int(rng.choice(options))
    n_minor = int(rng.integers(2, 6))
    for i in range(n_major * n_minor + 1):
        a = spec.min_angle_deg + spec.sweep_deg * i / (n_major * n_minor)
        major = i % n_minor == 0
        p1 = polar_to_xy(cx, cy, r * (0.80 if major else 0.86), a)
        p2 = polar_to_xy(cx, cy, r * 0.93, a)
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (20, 20, 20),
                 3 if major else 1, cv2.LINE_AA)
        if major:
            label = _fmt(spec.min_value + spec.span * i / (n_major * n_minor))
            scale = r / 300
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tx, ty = polar_to_xy(cx, cy, r * 0.68, a)
            cv2.putText(img, label, (int(tx - tw / 2), int(ty + th / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 1, cv2.LINE_AA)

    unit = str(rng.choice(["bar", "psi", "kPa", "°C"]))
    (tw, _), _ = cv2.getTextSize(unit, cv2.FONT_HERSHEY_SIMPLEX, r / 250, 1)
    cv2.putText(img, unit, (int(cx - tw / 2), int(cy + r * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, r / 250, (40, 40, 40), 1, cv2.LINE_AA)

    # Needle: tapered polygon from a short tail through the hub to the tip.
    a = spec.value_to_angle(value)
    color = (0, 0, int(rng.integers(150, 220))) if rng.random() < 0.5 else (15, 15, 15)
    half_w = r * float(rng.uniform(0.025, 0.04))
    tip = polar_to_xy(cx, cy, r * float(rng.uniform(0.75, 0.85)), a)
    tail = polar_to_xy(cx, cy, r * 0.15, a + 180)
    left = polar_to_xy(cx, cy, half_w, a - 90)
    right = polar_to_xy(cx, cy, half_w, a + 90)
    pts = np.array([tip, left, tail, right], dtype=np.float32)
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], color, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(r * 0.07), (30, 30, 30), -1, cv2.LINE_AA)


def _background(size: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.integers(40, 200, size=3)
    img = np.empty((size, size, 3), np.uint8)
    img[:] = base
    for _ in range(int(rng.integers(3, 9))):  # pipes, panels, cables
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        if rng.random() < 0.5:
            p1 = tuple(int(c) for c in rng.integers(0, size, size=2))
            p2 = tuple(int(c) for c in rng.integers(0, size, size=2))
            cv2.rectangle(img, p1, p2, color, -1)
        else:
            p1 = tuple(int(c) for c in rng.integers(0, size, size=2))
            p2 = tuple(int(c) for c in rng.integers(0, size, size=2))
            cv2.line(img, p1, p2, color, int(rng.integers(2, 15)), cv2.LINE_AA)
    return img


def render_gauge(
    rng: np.random.Generator,
    value: float | None = None,
    spec: DialSpec | None = None,
    size: int = 480,
    realistic: bool = True,
) -> tuple[np.ndarray, GroundTruth]:
    """Render one gauge image (BGR uint8) and its ground truth.

    realistic=False gives a clean, frontal image; realistic=True adds clutter, mild
    perspective, blur, sensor noise, lighting changes and occasional glare.
    """
    spec = spec or DIAL_TYPES[int(rng.integers(len(DIAL_TYPES)))]
    if value is None:
        value = float(rng.uniform(spec.min_value, spec.max_value))

    img = _background(size, rng) if realistic else np.full((size, size, 3), 128, np.uint8)
    r = size * float(rng.uniform(0.28, 0.40)) if realistic else size * 0.38
    margin = r * 1.1
    cx = float(rng.uniform(margin, size - margin)) if realistic else size / 2
    cy = float(rng.uniform(margin, size - margin)) if realistic else size / 2
    _draw_dial(img, cx, cy, r, spec, value, rng)

    if realistic:
        # Mild perspective (camera not exactly frontal).
        j = size * 0.035
        src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        dst = src + rng.uniform(-j, j, size=(4, 2)).astype(np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, H, (size, size), borderMode=cv2.BORDER_REPLICATE)
        c = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0, 0]
        cx, cy = float(c[0]), float(c[1])

        # Lighting and glare.
        img = cv2.convertScaleAbs(img, alpha=float(rng.uniform(0.6, 1.3)), beta=float(rng.uniform(-40, 40)))
        if rng.random() < 0.3:
            glare = np.zeros_like(img)
            gx, gy = (int(v) for v in rng.integers(0, size, size=2))
            axes = (int(rng.integers(20, 80)), int(rng.integers(10, 40)))
            cv2.ellipse(glare, (gx, gy), axes, float(rng.uniform(0, 180)), 0, 360, (255, 255, 255), -1)
            glare = cv2.GaussianBlur(glare, (0, 0), 15)
            img = cv2.addWeighted(img, 1.0, glare, 0.6, 0)

        # Optics and sensor.
        sigma = float(rng.uniform(0, 2.0))
        if sigma > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        noise = rng.normal(0, float(rng.uniform(0, 12)), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, GroundTruth((cx, cy), r, value, spec)

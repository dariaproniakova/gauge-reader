"""Dial calibration: the mapping between needle angle and gauge value.

Angle convention used throughout the package: degrees, 0° = 12 o'clock (pointing up),
increasing clockwise as seen in the image.
"""

from dataclasses import dataclass
import math

from .geometry import angle_to_value


@dataclass(frozen=True)
class DialSpec:
    """Scale of one gauge: where the scale starts/ends and the values at its ends."""

    min_angle_deg: float
    max_angle_deg: float
    min_value: float
    max_value: float

    @property
    def sweep_deg(self) -> float:
        """Angular length of the scale, measured clockwise from min to max."""
        return (self.max_angle_deg - self.min_angle_deg) % 360 or 360.0

    @property
    def span(self) -> float:
        """Full-scale range in value units."""
        return self.max_value - self.min_value

    def value_to_angle(self, value: float) -> float:
        frac = (value - self.min_value) / self.span
        return (self.min_angle_deg + frac * self.sweep_deg) % 360

    def angle_to_value(self, angle_deg: float) -> float:
        return angle_to_value(
            angle_deg, self.min_angle_deg, self.max_angle_deg, self.min_value, self.max_value
        )

    def in_sweep(self, angle_deg: float, margin_deg: float = 0.0) -> bool:
        """True if the angle lies on the scale (optionally extended by a margin)."""
        offset = (angle_deg - self.min_angle_deg + margin_deg) % 360
        return offset <= self.sweep_deg + 2 * margin_deg


def polar_to_xy(cx: float, cy: float, radius: float, angle_deg: float) -> tuple[float, float]:
    """Image coordinates of the point at `radius` from the centre in direction `angle_deg`."""
    a = math.radians(angle_deg)
    return cx + radius * math.sin(a), cy - radius * math.cos(a)


def xy_to_angle(dx: float, dy: float) -> float:
    """Angle (package convention) of the image-space vector (dx, dy)."""
    return math.degrees(math.atan2(dx, -dy)) % 360

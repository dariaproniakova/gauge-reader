"""Geometry helpers: converting a needle angle into a gauge reading."""


def angle_to_value(
    angle_deg: float,
    min_angle_deg: float,
    max_angle_deg: float,
    min_value: float,
    max_value: float,
) -> float:
    """Linearly map a needle angle to a gauge value.

    Angles are measured clockwise in degrees (image convention). The scale runs
    clockwise from ``min_angle_deg`` to ``max_angle_deg`` and may wrap past 360°,
    as on a typical 270° dial.
    """
    sweep = (max_angle_deg - min_angle_deg) % 360 or 360
    offset = (angle_deg - min_angle_deg) % 360
    if offset > sweep:
        # Needle is in the dead zone: snap to the nearer scale end.
        offset = sweep if offset - sweep < 360 - offset else 0.0
    return min_value + (offset / sweep) * (max_value - min_value)

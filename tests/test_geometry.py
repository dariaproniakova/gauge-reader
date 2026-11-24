import pytest

from gauge_reader import angle_to_value

# Typical 270° dial: 0 at 225° (bottom-left), max at 135° (bottom-right), clockwise.
DIAL = dict(min_angle_deg=225, max_angle_deg=135, min_value=0, max_value=10)


@pytest.mark.parametrize(
    "angle, expected",
    [(225, 0.0), (315, 10 / 3), (0, 5.0), (45, 20 / 3), (135, 10.0)],
)
def test_angle_to_value_on_270_dial(angle, expected):
    assert angle_to_value(angle, **DIAL) == pytest.approx(expected)


def test_dead_zone_snaps_to_nearest_end():
    assert angle_to_value(150, **DIAL) == pytest.approx(10.0)
    assert angle_to_value(210, **DIAL) == pytest.approx(0.0)

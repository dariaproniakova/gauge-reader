import numpy as np
import pytest

from gauge_reader.dial import DialSpec, polar_to_xy, xy_to_angle
from gauge_reader.reader import find_dial, read_gauge
from gauge_reader.synthetic import DIAL_TYPES, render_gauge


def test_polar_roundtrip():
    for a in [0, 45, 90, 180, 225, 359]:
        x, y = polar_to_xy(100, 100, 50, a)
        assert xy_to_angle(x - 100, y - 100) == pytest.approx(a % 360, abs=1e-9)


def test_dial_spec_value_angle_roundtrip():
    spec = DialSpec(225, 135, 0, 16)
    for v in [0, 4, 8, 12, 16]:
        assert spec.angle_to_value(spec.value_to_angle(v)) == pytest.approx(v)


def test_find_dial_on_clean_image():
    img, gt = render_gauge(np.random.default_rng(0), realistic=False)
    dial = find_dial(img)
    assert dial is not None
    assert np.hypot(dial.cx - gt.center[0], dial.cy - gt.center[1]) < 0.05 * gt.radius


@pytest.mark.parametrize("spec", DIAL_TYPES)
def test_reads_clean_gauges_accurately(spec):
    rng = np.random.default_rng(1)
    for value in np.linspace(spec.min_value, spec.max_value, 7):
        img, gt = render_gauge(rng, value=float(value), spec=spec, realistic=False)
        reading = read_gauge(img, spec)
        assert reading is not None
        assert abs(reading.value - value) / spec.span < 0.02


def test_reads_realistic_gauges():
    rng = np.random.default_rng(2)
    errors = []
    for _ in range(40):
        img, gt = render_gauge(rng, realistic=True)
        reading = read_gauge(img, gt.spec)
        assert reading is not None
        errors.append(abs(reading.value - gt.value) / gt.spec.span)
    assert np.mean(errors) < 0.01
    assert np.max(errors) < 0.05

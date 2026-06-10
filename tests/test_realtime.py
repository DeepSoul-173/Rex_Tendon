"""Unit tests for rex_tendon.control.realtime (pure NumPy, no sim)."""

import time

import numpy as np
import pytest

from rex_tendon.control.realtime import (
    CommandShaper,
    LatestValue,
    RateClock,
    ShaperConfig,
)

DT = 1.0 / 60.0


# ── LatestValue ───────────────────────────────────────────────────────────────


def test_latest_value_empty():
    slot: LatestValue[int] = LatestValue()
    value, age = slot.get()
    assert value is None and age is None
    assert slot.is_stale(max_age=10.0)


def test_latest_value_freshness():
    slot: LatestValue[str] = LatestValue()
    slot.set("hello")
    value, age = slot.get()
    assert value == "hello"
    assert age is not None and age < 0.5
    assert not slot.is_stale(max_age=0.5)


def test_latest_value_goes_stale():
    slot: LatestValue[int] = LatestValue()
    slot.set(1)
    time.sleep(0.03)
    assert slot.is_stale(max_age=0.01)


def test_latest_value_keeps_newest():
    slot: LatestValue[int] = LatestValue()
    slot.set(1)
    slot.set(2)
    value, _ = slot.get()
    assert value == 2


# ── RateClock ─────────────────────────────────────────────────────────────────


def test_rate_clock_first_tick_returns_period():
    clock = RateClock(60.0)
    assert clock.tick() == pytest.approx(1.0 / 60.0)


def test_rate_clock_paces_loop():
    clock = RateClock(100.0)
    clock.tick()
    t0 = time.perf_counter()
    dts = [clock.tick() for _ in range(10)]
    elapsed = time.perf_counter() - t0
    # 10 ticks at 100 Hz ~ 0.1 s; allow generous scheduler slack on Windows.
    assert elapsed >= 0.08
    assert all(dt > 0.0 for dt in dts)


# ── CommandShaper ─────────────────────────────────────────────────────────────


def test_shaper_zero_inside_deadzone():
    shaper = CommandShaper(ShaperConfig(deadzone=0.1, max_rate=None))
    for _ in range(5):
        out = shaper.shape(np.array([0.05, 0.05]), DT)
    assert np.allclose(out, 0.0)


def test_shaper_output_clamped():
    shaper = CommandShaper(
        ShaperConfig(gain=10.0, out_min=-1.0, out_max=1.0, max_rate=None)
    )
    out = 0.0
    for _ in range(300):
        out = shaper.shape(0.9, DT)
    assert out <= 1.0
    assert out == pytest.approx(1.0, abs=1e-2)


def test_shaper_respects_slew_limit():
    shaper = CommandShaper(
        ShaperConfig(min_cutoff=100.0, beta=10.0, max_rate=1.0)
    )
    prev = shaper.shape(0.0, DT)
    out = shaper.shape(1.0, DT)
    assert abs(out - prev) <= 1.0 * DT + 1e-9


def test_shaper_scalar_in_scalar_out():
    shaper = CommandShaper(ShaperConfig())
    assert isinstance(shaper.shape(0.5, DT), float)


def test_shaper_reset():
    shaper = CommandShaper(ShaperConfig(max_rate=5.0))
    for _ in range(10):
        shaper.shape(1.0, DT)
    shaper.reset()
    # After reset the first sample passes through (no slew from old state).
    assert shaper.shape(0.0, DT) == pytest.approx(0.0)

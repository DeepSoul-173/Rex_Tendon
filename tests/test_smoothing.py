"""Unit tests for rex_tendon.control.smoothing (pure NumPy, no sim)."""

import numpy as np
import pytest

from rex_tendon.control.smoothing import (
    Deadzone,
    ExponentialSmoother,
    HoldToTrigger,
    Hysteresis,
    OneEuroFilter,
    SlewRateLimiter,
)

DT = 1.0 / 60.0


# ── OneEuroFilter ─────────────────────────────────────────────────────────────


def test_one_euro_first_sample_passes_through():
    f = OneEuroFilter()
    assert f.filter(0.7, DT) == pytest.approx(0.7)


def test_one_euro_converges_to_constant():
    f = OneEuroFilter(min_cutoff=1.0)
    y = 0.0
    for _ in range(600):  # 10 s of constant input
        y = f.filter(1.0, DT)
    assert y == pytest.approx(1.0, abs=1e-3)


def test_one_euro_attenuates_jitter():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.05, size=600)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = np.array([f.filter(0.5 + n, DT) for n in noise])
    assert np.std(out[100:]) < 0.5 * np.std(noise[100:])


def test_one_euro_beta_reduces_lag_on_fast_motion():
    # Track a fast ramp; higher beta should stay closer to the input.
    ramp = np.linspace(0.0, 1.0, 60)  # full sweep in 1 s
    slow = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    fast = OneEuroFilter(min_cutoff=0.5, beta=1.0)
    err_slow = err_fast = 0.0
    for x in ramp:
        err_slow += abs(x - slow.filter(x, DT))
        err_fast += abs(x - fast.filter(x, DT))
    assert err_fast < err_slow


def test_one_euro_vector_input():
    f = OneEuroFilter()
    out = f.filter(np.array([0.1, -0.2]), DT)
    assert isinstance(out, np.ndarray) and out.shape == (2,)
    out = f.filter(np.array([0.2, -0.1]), DT)
    assert out.shape == (2,)


def test_one_euro_reset():
    f = OneEuroFilter()
    f.filter(1.0, DT)
    f.reset()
    assert f.filter(5.0, DT) == pytest.approx(5.0)


# ── ExponentialSmoother ───────────────────────────────────────────────────────


def test_ema_tau_step_response():
    # After tau seconds, a step should be ~63% absorbed regardless of dt.
    for dt in (1.0 / 30.0, 1.0 / 120.0):
        s = ExponentialSmoother(tau=0.1)
        s.filter(0.0, dt)
        y = 0.0
        t = 0.0
        while t < 0.1 - 1e-9:
            y = s.filter(1.0, dt)
            t += dt
        assert 0.5 < y < 0.75


def test_ema_zero_tau_is_passthrough():
    s = ExponentialSmoother(tau=0.0)
    s.filter(0.0, DT)
    assert s.filter(1.0, DT) == pytest.approx(1.0)


# ── Deadzone ──────────────────────────────────────────────────────────────────


def test_deadzone_zero_inside():
    dz = Deadzone(radius=0.1)
    assert dz.apply(0.05) == 0.0
    assert np.allclose(dz.apply(np.array([0.05, 0.05])), 0.0)


def test_deadzone_continuous_at_boundary():
    dz = Deadzone(radius=0.1)
    just_outside = dz.apply(0.10001)
    assert abs(just_outside) < 0.01  # no jump


def test_deadzone_full_scale_preserved():
    dz = Deadzone(radius=0.1, full_scale=1.0)
    assert dz.apply(1.0) == pytest.approx(1.0)
    v = dz.apply(np.array([0.6, 0.8]))  # magnitude 1.0
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_deadzone_preserves_direction():
    dz = Deadzone(radius=0.1)
    v = dz.apply(np.array([0.3, 0.4]))
    assert v[0] / v[1] == pytest.approx(0.75)


# ── SlewRateLimiter ───────────────────────────────────────────────────────────


def test_slew_caps_rate():
    lim = SlewRateLimiter(max_rate=1.0)
    lim.step(0.0, DT)
    y = lim.step(10.0, DT)
    assert y == pytest.approx(1.0 * DT)


def test_slew_reaches_target():
    lim = SlewRateLimiter(max_rate=2.0)
    lim.step(0.0, DT)
    y = 0.0
    for _ in range(120):  # 2 s at 2 units/s -> covers 4 units
        y = lim.step(1.0, DT)
    assert y == pytest.approx(1.0)


def test_slew_vector():
    lim = SlewRateLimiter(max_rate=1.0)
    lim.step(np.zeros(3), DT)
    y = lim.step(np.array([10.0, -10.0, 0.0]), DT)
    assert np.allclose(y, [DT, -DT, 0.0])


# ── HoldToTrigger ─────────────────────────────────────────────────────────────


def test_hold_fires_once_after_hold_time():
    h = HoldToTrigger(hold_time=0.1)
    fired = [h.update(True, DT) for _ in range(12)]  # 0.2 s held
    assert sum(fired) == 1
    assert fired.index(True) in (5, 6)  # first tick at/after 0.1 s of hold


def test_hold_resets_on_release():
    h = HoldToTrigger(hold_time=0.1)
    for _ in range(3):
        h.update(True, DT)
    h.update(False, DT)
    assert h.progress == 0.0
    fired = [h.update(True, DT) for _ in range(12)]
    assert sum(fired) == 1  # can fire again after release


def test_hold_progress():
    h = HoldToTrigger(hold_time=0.1)
    for _ in range(3):
        h.update(True, DT)
    assert 0.0 < h.progress < 1.0


# ── Hysteresis ────────────────────────────────────────────────────────────────


def test_hysteresis_engage_and_rearm():
    hy = Hysteresis(engage_below=0.35, release_above=0.55)
    assert not hy.update(0.5)  # between thresholds, stays off
    assert hy.update(0.3)  # engages
    assert hy.update(0.45)  # stays engaged between thresholds
    assert not hy.update(0.6)  # releases
    assert not hy.update(0.45)  # stays off until below engage threshold
    assert hy.update(0.2)

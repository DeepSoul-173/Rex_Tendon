"""Reusable real-time signal conditioning for teleoperation.

Every filter here is dt-aware: behaviour is specified in seconds, not frames,
so control feel is identical at 30 fps and 60 fps. This replaces the
frame-rate-dependent EMA / per-frame rate caps / frame-count gesture timers
that are currently inlined in hand_sim_controller.py.

Contents
--------
  OneEuroFilter      : adaptive low-pass — minimal jitter at rest, minimal lag
                       in motion (Casiez et al., CHI 2012). The right default
                       for hand-tracking cursors.
  ExponentialSmoother: plain EMA with a time-constant (tau seconds) instead of
                       a per-frame alpha.
  Deadzone           : radial dead-zone with continuous rescaling (no output
                       jump at the dead-zone boundary).
  SlewRateLimiter    : caps output change in units/second.
  HoldToTrigger      : "hold gesture for T seconds" debouncer with progress.
  Hysteresis         : two-threshold boolean state (e.g. pinch lock/re-arm).

All numeric filters accept floats or NumPy arrays and preserve the input kind.
None of this imports MuJoCo / MediaPipe / cv2 — safe to unit-test anywhere.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Signal = Union[float, np.ndarray]


def _as_array(x: Signal) -> tuple[np.ndarray, bool]:
    """Return (float64 array, was_scalar)."""
    arr = np.asarray(x, dtype=np.float64)
    return arr, arr.ndim == 0


def _restore(arr: np.ndarray, was_scalar: bool) -> Signal:
    return float(arr) if was_scalar else arr


# ── One Euro filter ───────────────────────────────────────────────────────────


def _smoothing_factor(dt: float, cutoff: Signal) -> Signal:
    """Map a cutoff frequency (Hz) to an EMA alpha for this timestep."""
    r = 2.0 * math.pi * np.asarray(cutoff, dtype=np.float64) * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """Adaptive low-pass filter for noisy interactive signals.

    The cutoff frequency rises with signal speed: a still hand gets heavy
    smoothing (jitter removed), a fast-moving hand gets light smoothing
    (no perceptible lag).

    Tuning (per Casiez et al.):
      min_cutoff : lower → smoother at rest, more lag on slow drift.
                   Start ~1.0 Hz for cursors.
      beta       : higher → less lag during fast motion. Start 0.0 and raise
                   until fast moves feel direct (~0.05–0.5 typical).
      d_cutoff   : cutoff for the internal speed estimate; 1.0 Hz is fine.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        if min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError("min_cutoff and d_cutoff must be > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def filter(self, x: Signal, dt: float) -> Signal:
        """Filter one sample taken dt seconds after the previous one."""
        arr, was_scalar = _as_array(x)

        if self._x_prev is None or dt <= 0.0:
            self._x_prev = arr.copy()
            self._dx_prev = np.zeros_like(arr)
            return _restore(arr, was_scalar)

        # Smoothed speed estimate.
        dx = (arr - self._x_prev) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Speed-adaptive cutoff, then the actual low-pass.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = a * arr + (1.0 - a) * self._x_prev

        self._x_prev = np.asarray(x_hat, dtype=np.float64)
        self._dx_prev = np.asarray(dx_hat, dtype=np.float64)
        return _restore(self._x_prev, was_scalar)


# ── Exponential smoother (time-constant EMA) ──────────────────────────────────


class ExponentialSmoother:
    """EMA specified by a time constant tau (seconds), not a per-frame alpha.

    After tau seconds the output has covered ~63% of a step change, regardless
    of frame rate. tau=0 disables smoothing (pass-through).
    """

    def __init__(self, tau: float = 0.05):
        if tau < 0.0:
            raise ValueError("tau must be >= 0")
        self.tau = float(tau)
        self._y: np.ndarray | None = None

    def reset(self, value: Signal | None = None) -> None:
        self._y = None if value is None else _as_array(value)[0].copy()

    def filter(self, x: Signal, dt: float) -> Signal:
        arr, was_scalar = _as_array(x)
        if self._y is None or dt <= 0.0 or self.tau == 0.0:
            self._y = arr.copy()
            return _restore(arr, was_scalar)
        alpha = 1.0 - math.exp(-dt / self.tau)
        self._y = alpha * arr + (1.0 - alpha) * self._y
        return _restore(self._y, was_scalar)


# ── Dead-zone ─────────────────────────────────────────────────────────────────


class Deadzone:
    """Radial dead-zone with continuous rescaling.

    Inside the radius the output is exactly zero. Outside, magnitude is
    remapped from [radius, full_scale] to [0, full_scale], so the output is
    continuous at the boundary — no jump when the hand leaves the dead-zone
    (the current _map_cursor dead-zone jumps from 0 to gain*radius).
    """

    def __init__(self, radius: float, full_scale: float = 1.0):
        if not 0.0 <= radius < full_scale:
            raise ValueError("need 0 <= radius < full_scale")
        self.radius = float(radius)
        self.full_scale = float(full_scale)

    def apply(self, x: Signal) -> Signal:
        arr, was_scalar = _as_array(x)
        mag = float(np.linalg.norm(arr)) if arr.ndim else abs(float(arr))
        if mag <= self.radius:
            return _restore(np.zeros_like(arr), was_scalar)
        scaled = (
            (mag - self.radius)
            / (self.full_scale - self.radius)
            * self.full_scale
        )
        scaled = min(scaled, self.full_scale)
        return _restore(arr * (scaled / mag), was_scalar)


# ── Slew-rate limiter ─────────────────────────────────────────────────────────


class SlewRateLimiter:
    """Cap output change at max_rate units per second (per element).

    Frame-rate-independent replacement for the per-frame MAX_CTRL_STEP clip:
    at 60 fps a max_rate of 1.8 units/s equals the old 0.03 units/frame.
    """

    def __init__(self, max_rate: float):
        if max_rate <= 0.0:
            raise ValueError("max_rate must be > 0")
        self.max_rate = float(max_rate)
        self._y: np.ndarray | None = None

    def reset(self, value: Signal | None = None) -> None:
        self._y = None if value is None else _as_array(value)[0].copy()

    def step(self, target: Signal, dt: float) -> Signal:
        arr, was_scalar = _as_array(target)
        if self._y is None or dt <= 0.0:
            self._y = arr.copy()
            return _restore(arr, was_scalar)
        max_step = self.max_rate * dt
        self._y = self._y + np.clip(arr - self._y, -max_step, max_step)
        return _restore(self._y, was_scalar)


# ── Gesture debouncing ────────────────────────────────────────────────────────


class HoldToTrigger:
    """Fire once after a condition has been continuously true for hold_time s.

    Time-based replacement for frame counters (FIST_LOCK_FRAMES etc.).
    update() returns True exactly once per sustained hold; the condition must
    drop before the trigger can fire again. `progress` (0..1) drives HUD
    "charging" feedback.
    """

    def __init__(self, hold_time: float):
        if hold_time < 0.0:
            raise ValueError("hold_time must be >= 0")
        self.hold_time = float(hold_time)
        self._held = 0.0
        self._fired = False

    @property
    def progress(self) -> float:
        if self.hold_time == 0.0:
            return 1.0 if self._held > 0.0 else 0.0
        return min(self._held / self.hold_time, 1.0)

    def reset(self) -> None:
        self._held = 0.0
        self._fired = False

    def update(self, active: bool, dt: float) -> bool:
        if not active:
            self.reset()
            return False
        self._held += max(dt, 0.0)
        if not self._fired and self._held >= self.hold_time:
            self._fired = True
            return True
        return False


class Hysteresis:
    """Boolean state with separate engage/release thresholds on a scalar.

    Example (pinch lock): engage when pinch_ratio < 0.35, but only re-arm
    after it has risen above 0.55 — the existing _pinch_was_open pattern,
    made explicit and reusable.
    """

    def __init__(self, engage_below: float, release_above: float):
        if release_above <= engage_below:
            raise ValueError("release_above must be > engage_below")
        self.engage_below = float(engage_below)
        self.release_above = float(release_above)
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def reset(self) -> None:
        self._engaged = False

    def update(self, value: float) -> bool:
        if self._engaged:
            if value > self.release_above:
                self._engaged = False
        elif value < self.engage_below:
            self._engaged = True
        return self._engaged

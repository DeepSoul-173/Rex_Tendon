"""Low-latency teleoperation plumbing: freshness, pacing, command shaping.

Generalizes patterns currently inlined in hand_sim_controller.py so any
control mode (gesture, voice, trackpad) can reuse them:

  LatestValue   : thread-safe single-slot mailbox with staleness age — the
                  _CamThread "always read the newest sample, never block"
                  pattern, plus the stale-input check the controller is
                  missing (it reads cam_ts but never inspects it).
  RateClock     : fixed-rate loop pacing that reports the *measured* dt, which
                  is what the dt-aware filters in smoothing.py should be fed.
  CommandShaper : the full per-axis conditioning chain as one object:
                  dead-zone -> gain -> clamp -> One Euro -> slew limit.

No MuJoCo / MediaPipe / cv2 imports.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from .smoothing import Deadzone, OneEuroFilter, SlewRateLimiter

T = TypeVar("T")
Signal = Union[float, np.ndarray]


# ── Freshness-aware mailbox ───────────────────────────────────────────────────


class LatestValue(Generic[T]):
    """Thread-safe slot holding the most recent value and its timestamp.

    Producer calls set(); consumer calls get() and receives (value, age_s).
    age_s is None until the first set(). Use is_stale() as a safety gate:
    a teleop loop should hold position (or go neutral) when its input source
    stalls, rather than keep acting on a frozen sample.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._stamp: Optional[float] = None

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._stamp = time.perf_counter()

    def get(self) -> tuple[Optional[T], Optional[float]]:
        with self._lock:
            if self._stamp is None:
                return None, None
            return self._value, time.perf_counter() - self._stamp

    def is_stale(self, max_age: float) -> bool:
        """True if no value yet, or the newest value is older than max_age s."""
        _, age = self.get()
        return age is None or age > max_age


# ── Loop pacing ───────────────────────────────────────────────────────────────


class RateClock:
    """Pace a loop at rate_hz and report the measured dt each tick.

    Usage:
        clock = RateClock(60.0)
        while running:
            dt = clock.tick()      # sleeps off any surplus, returns real dt
            ...use dt for filters...

    Overruns don't accumulate: after a slow iteration the schedule resets to
    "now" instead of bursting to catch up.
    """

    def __init__(self, rate_hz: float):
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        self.period = 1.0 / float(rate_hz)
        self._next: Optional[float] = None
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._next is not None:
            sleep_for = self._next - now
            if sleep_for > 0.0:
                time.sleep(sleep_for)
                now = time.perf_counter()
        dt = self.period if self._last is None else now - self._last
        self._last = now
        # Schedule from "now" so an overrun doesn't cause a catch-up burst.
        self._next = now + self.period
        return dt


# ── Command shaping ───────────────────────────────────────────────────────────


@dataclass
class ShaperConfig:
    """Per-channel conditioning parameters (units: seconds and signal units).

    deadzone   : input radius mapped to zero (continuous rescale outside).
    gain       : multiplier applied after the dead-zone.
    out_min/max: hard clamp after gain (per element).
    min_cutoff : One Euro smoothing at rest (Hz). Lower = smoother.
    beta       : One Euro speed coefficient. Higher = less lag when fast.
    max_rate   : output slew limit in units/second (None = unlimited).
    """

    deadzone: float = 0.0
    gain: float = 1.0
    out_min: float = -1.0
    out_max: float = 1.0
    min_cutoff: float = 1.0
    beta: float = 0.1
    max_rate: Optional[float] = None


class CommandShaper:
    """Dead-zone -> gain -> clamp -> One Euro -> slew, as one dt-aware object.

    This is the cursor/yaw conditioning chain from hand_sim_controller.py in
    reusable form: feed it raw normalized input and the loop's measured dt,
    get a bounded, smooth, rate-limited command back.
    """

    def __init__(self, config: ShaperConfig):
        self.config = config
        self._deadzone = (
            Deadzone(config.deadzone) if config.deadzone > 0.0 else None
        )
        self._filter = OneEuroFilter(
            min_cutoff=config.min_cutoff, beta=config.beta
        )
        self._slew = (
            SlewRateLimiter(config.max_rate)
            if config.max_rate is not None
            else None
        )

    def reset(self) -> None:
        self._filter.reset()
        if self._slew is not None:
            self._slew.reset()

    def shape(self, raw: Signal, dt: float) -> Signal:
        x = self._deadzone.apply(raw) if self._deadzone is not None else raw
        x = np.clip(
            np.asarray(x, dtype=np.float64) * self.config.gain,
            self.config.out_min,
            self.config.out_max,
        )
        if np.ndim(raw) == 0:
            x = float(x)
        x = self._filter.filter(x, dt)
        if self._slew is not None:
            x = self._slew.step(x, dt)
        return x

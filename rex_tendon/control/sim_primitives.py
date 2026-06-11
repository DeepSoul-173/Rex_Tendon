"""Scripted movement primitives for the simulated tentacle arm.

The execution backend for high-level commands (voice, scripted demos,
benchmarks): approach → grasp → carry → place → release, built on the same
cursor → tendon-length mapping and assisted-grasp mechanics the hand
controller uses, but driven by feedback control instead of a webcam.

    arm = SimArm("rex_assets/rex_simulation/pick_and_place_scene.xml", viewer=True)
    red = arm.find_object(color="red")
    arm.grasp(red)
    arm.place_at(arm.place_zone_position())

Headless by default (viewer optional), no camera / MediaPipe / RL imports —
usable from tests, the voice runner, and benchmark scripts alike.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from ..perception.scene_objects import (
    SceneObject,
    available_colors,
    discover_objects,
    find_by_color,
    nearest_object,
)
from .geometry import convert_2d_cursor_to_target_lengths
from .smoothing import SlewRateLimiter

logger = logging.getLogger(__name__)

# Arm geometry (matches hand_sim_controller's constants for the same scenes).
NEUTRAL_LEN = 0.23
GRASP_DIST = 0.055  # tip-to-object distance that counts as a secure hold
# Carry depth: the tip's contact sphere has radius 0.008; the carried cube
# (half-height 0.01) must hang clear of it or every-substep teleporting
# re-creates penetration and the contact impulses shake the whole arm
# (observed: arm never settles while carrying, slingshots the cube on release).
CARRY_OFFSET = np.array([0.0, 0.0, -0.022], dtype=np.float64)

# Cursor feedback control. Measured facts that shape this design:
# 1. The arm is underdamped (sways >1 s after a move): positioning must be
#    iterative aim -> glide -> settle -> re-aim on SETTLED measurements.
# 2. The cursor->tip map is folded (small cursor tilts the standing arm high;
#    large cursor coils it down to working height) and its direction rotates
#    with bend magnitude — a local linear (Newton/Broyden) model flails when
#    crossing quadrants.
# 3. The natural parametrization for the spiral is POLAR: cursor azimuth
#    steers the bend direction, cursor magnitude sets the reach radius. Both
#    relations are monotone on the coil branch, so two decoupled 1-D feedback
#    loops converge where a 2x2 scheme could not. Calibrated per scene by
#    probing N bend directions at startup.
# Radius control cascades: cursor magnitude first, then baseline (tendon
# extension) once magnitude saturates — measured settled reach tops out at
# r ~ 0.07 m (baseline slope ~0.18 m radius per m baseline at full bend).
AZ_GAIN = 0.5  # cursor-azimuth correction per rad of tip-azimuth error
AZ_STEP_MAX = 0.6  # rad cap per aim iteration (damps the az/coil coupling)
R_GAIN = 5.0  # cursor-magnitude correction per meter of radius error
M_MIN, M_MAX = 0.6, 1.45  # cursor magnitude range (coil branch .. box corners)
BASELINE_MIN, BASELINE_MAX = 0.16, 0.30
R2B_SLOPE = 0.18  # m of radius per m of baseline (measured at full bend)
BASELINE_STEP_MAX = 0.04  # baseline change cap per aim iteration
R_REACH = 0.069  # max settled tip radius (measured; beyond it = swing-only)
PLACE_TOL = 0.06  # the scene's place-zone disc radius — its own tolerance
CURSOR_RATE = 2.5  # max cursor change per second (slew during glide)
GLIDE_TICKS = 30  # ticks spent slewing toward each new cursor target
SETTLE_TICKS = 70  # ~0.85 s of settling before each measurement
WORK_Z = 0.045  # tip height the coil branch allows at grasping radii
SETTLE_STEPS = 30  # post-release physics steps
_CAL_MAG = 0.85  # calibration probe magnitude (coil branch)
_CAL_AZIMUTHS = 8  # bend directions probed at startup
_CAL_TICKS = 100  # ticks to settle each calibration probe


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class SimArm:
    """Feedback-controlled scripted session on the pick-and-place scene."""

    def __init__(
        self,
        xml_path: str = "rex_assets/rex_simulation/pick_and_place_scene.xml",
        viewer: bool = False,
        realtime: bool = False,
        arrange_objects: bool = False,
        arrange_seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime

        self.act_low = self.model.actuator_ctrlrange[:3, 0]
        self.act_high = self.model.actuator_ctrlrange[:3, 1]
        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_center"
        )
        self.place_zone_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "place_zone"
        )
        self.objects = discover_objects(self.model)
        logger.info(
            "SimArm: %d objects discovered: %s",
            len(self.objects),
            {o.name: o.color for o in self.objects},
        )

        # Control state
        self.cursor = np.zeros(2, dtype=np.float64)
        self.baseline = float(NEUTRAL_LEN)
        self.grasped: Optional[SceneObject] = None
        # Per-control-tick slew limit keeps scripted motion physical.
        self._dt = float(self.model.opt.timestep) * 6  # 6 substeps per tick
        self._cursor_slew = SlewRateLimiter(CURSOR_RATE)
        self._cursor_slew.reset(self.cursor)

        mujoco.mj_forward(self.model, self.data)
        self._write_ctrl()

        # Self-calibrate the polar cursor->tip map, then restore a clean
        # scene. The viewer attaches afterwards so calibration is invisible.
        self._viewer = None
        self._calibrate_polar()
        # The XML's default object layout spreads across the whole table; most
        # of it is beyond the settled reach (~0.069 m). Arranging pulls every
        # object into the workspace so voice commands can act on all of them.
        if arrange_objects:
            self._arrange_objects(arrange_seed)
        if viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ── Low-level ──────────────────────────────────────────────────────────────

    def tip_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()

    @staticmethod
    def _reachable_xy(xy: np.ndarray) -> np.ndarray:
        """Clamp an XY target onto the settled-reach disc.

        Aiming beyond R_REACH can never satisfy move_to's tolerance: the loop
        burns its iterations swinging and ends at a random phase (observed:
        releasing there flings the carried cube). Clamping converges to a calm
        settled pose at maximum reach along the target's direction.
        """
        r = float(np.linalg.norm(xy))
        if r <= R_REACH:
            return xy
        return xy * (R_REACH / r)

    def place_zone_position(self) -> np.ndarray:
        return self.data.site_xpos[self.place_zone_site_id].copy()

    def scene_colors(self) -> set[str]:
        return available_colors(self.objects)

    def find_object(
        self, color: Optional[str] = None, shape: Optional[str] = None
    ) -> Optional[SceneObject]:
        if color is not None:
            return find_by_color(self.objects, color, shape)
        obj, _ = nearest_object(self.objects, self.data, self.tip_position())
        return obj

    def _write_ctrl(self) -> None:
        lengths = convert_2d_cursor_to_target_lengths(
            self.cursor.astype(np.float32),
            np.full(3, self.baseline, dtype=np.float32),
            self.act_low,
            self.act_high,
            1.0,
        )
        self.data.ctrl[:3] = lengths

    def _calibrate_polar(self) -> None:
        """Probe N bend directions; record settled tip azimuth/radius for each.

        Yields the cursor-azimuth -> tip-azimuth correspondence (and the sign
        of that relation) used to seed and steer the polar feedback in
        move_to. Per-scene, no hardcoded signs or rotations.
        """
        az_cursor, az_tip, r_tip = [], [], []
        for k in range(_CAL_AZIMUTHS):
            az = -np.pi + 2.0 * np.pi * k / _CAL_AZIMUTHS
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.cursor = _CAL_MAG * np.array([np.cos(az), np.sin(az)])
            self._cursor_slew.reset(self.cursor)
            self.step(_CAL_TICKS)
            tip = self.tip_position()
            az_cursor.append(az)
            az_tip.append(float(np.arctan2(tip[1], tip[0])))
            r_tip.append(float(np.linalg.norm(tip[:2])))
        self._cal_az_cursor = np.array(az_cursor)
        self._cal_az_tip = np.array(az_tip)
        self._cal_r_tip = np.array(r_tip)
        # Sign of d(tip azimuth)/d(cursor azimuth): majority over probe pairs.
        diffs = [
            _wrap_angle(self._cal_az_tip[(i + 1) % _CAL_AZIMUTHS] - self._cal_az_tip[i])
            for i in range(_CAL_AZIMUTHS)
        ]
        self._az_sign = 1.0 if float(np.median(diffs)) >= 0.0 else -1.0
        # Restore a pristine scene + neutral pose.
        mujoco.mj_resetData(self.model, self.data)
        self.cursor = np.zeros(2, dtype=np.float64)
        self._cursor_slew.reset(self.cursor)
        self._write_ctrl()
        mujoco.mj_forward(self.model, self.data)

    def _arrange_objects(self, seed: int = 0) -> None:
        """Respawn every object at a random collision-free reachable spot."""
        rng = np.random.default_rng(seed)
        placed: list[np.ndarray] = []
        for obj in self.objects:
            xy = None
            for _ in range(80):
                az = rng.uniform(-np.pi, np.pi)
                r = rng.uniform(0.045, 0.066)
                candidate = np.array([r * np.cos(az), r * np.sin(az)])
                if all(np.linalg.norm(candidate - p) >= 0.05 for p in placed):
                    xy = candidate
                    break
            if xy is None:
                continue  # workspace full — leave this object where it is
            placed.append(xy)
            jnt = self.model.body_jntadr[obj.body_id]
            if jnt < 0:
                continue
            adr = self.model.jnt_qposadr[jnt]
            self.data.qpos[adr : adr + 3] = [xy[0], xy[1], obj.half_height + 0.001]
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            vadr = self.model.jnt_dofadr[jnt]
            if vadr >= 0:
                self.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.step(20)  # settle

    def _seed_cursor_for(self, target_xy: np.ndarray) -> tuple[float, float]:
        """Initial (cursor azimuth, magnitude) for a target from calibration."""
        az_t = float(np.arctan2(target_xy[1], target_xy[0]))
        errs = np.abs([_wrap_angle(a - az_t) for a in self._cal_az_tip])
        k = int(np.argmin(errs))
        r_cal = max(float(self._cal_r_tip[k]), 1e-3)
        mag = float(
            np.clip(_CAL_MAG * np.linalg.norm(target_xy) / r_cal, M_MIN, M_MAX)
        )
        return float(self._cal_az_cursor[k]), mag

    def _hold_grasped(self) -> None:
        """Assisted carry, mirroring the hand controller's mechanics."""
        if self.grasped is None:
            return
        bid = self.grasped.body_id
        jnt_adr = self.model.body_jntadr[bid]
        if jnt_adr < 0:
            return
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        qvel_adr = self.model.jnt_dofadr[jnt_adr]
        self.data.qpos[qpos_adr : qpos_adr + 3] = self.tip_position() + CARRY_OFFSET
        if qvel_adr >= 0:
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mass = float(self.model.body_mass[bid])
        self.data.xfrc_applied[bid, :3] = [0.0, 0.0, mass * 9.81]

    def step(self, n: int = 1) -> None:
        """Advance the sim n control ticks (6 physics substeps each)."""
        for _ in range(n):
            self._write_ctrl()
            for _ in range(6):
                if self.grasped is not None:
                    self._hold_grasped()
                mujoco.mj_step(self.model, self.data)
            if self.grasped is not None:
                self._hold_grasped()
                mujoco.mj_forward(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
                if self.realtime:
                    time.sleep(self._dt)

    # ── Primitives ─────────────────────────────────────────────────────────────

    def move_to(
        self,
        target: np.ndarray,
        tol: float = 0.02,
        z_tol: float = 0.04,
        max_iters: int = 12,
    ) -> bool:
        """Iterative aim-and-settle positioning of the tip at a world point.

        Each iteration computes a cursor correction from the SETTLED tip error
        through the calibrated inverse map, glides there slew-limited, and
        waits for the oscillation to die before re-measuring. XY is controlled;
        Z is dome-dependent and only checked against its looser tolerance.
        """
        target = np.asarray(target, dtype=np.float64)
        r_t = float(np.linalg.norm(target[:2]))
        az_t = float(np.arctan2(target[1], target[0]))
        az_c, mag = self._seed_cursor_for(target[:2])

        for _ in range(max_iters):
            tip = self.tip_position()
            err_xy = target[:2] - tip[:2]
            if (
                float(np.linalg.norm(err_xy)) <= tol
                and abs(target[2] - tip[2]) <= z_tol
            ):
                return True

            r_now = float(np.linalg.norm(tip[:2]))
            r_err = r_t - r_now
            if r_now > 0.015:  # tip azimuth is well-defined away from the base
                az_now = float(np.arctan2(tip[1], tip[0]))
                az_step = AZ_GAIN * self._az_sign * _wrap_angle(az_t - az_now)
                az_c = _wrap_angle(
                    az_c + float(np.clip(az_step, -AZ_STEP_MAX, AZ_STEP_MAX))
                )
            # Radius cascade: magnitude first; baseline once magnitude saturates.
            new_mag = float(np.clip(mag + R_GAIN * r_err, M_MIN, M_MAX))
            if new_mag == mag and abs(r_err) > tol * 0.5:
                bl_step = float(
                    np.clip(r_err / R2B_SLOPE, -BASELINE_STEP_MAX, BASELINE_STEP_MAX)
                )
                self.baseline = float(
                    np.clip(self.baseline + bl_step, BASELINE_MIN, BASELINE_MAX)
                )
            mag = new_mag

            desired = np.clip(
                mag * np.array([np.cos(az_c), np.sin(az_c)]), -1.0, 1.0
            )
            for _ in range(GLIDE_TICKS):
                self.cursor = np.asarray(self._cursor_slew.step(desired, self._dt))
                self.step()
            self.step(SETTLE_TICKS)

        tip = self.tip_position()
        return (
            float(np.linalg.norm(target[:2] - tip[:2])) <= tol
            and abs(target[2] - tip[2]) <= z_tol
        )

    def grasp(self, obj: Optional[SceneObject]) -> bool:
        """Drive the tip over the object and lock the assisted grasp.

        The dome forbids approaching from above at grasping radii, so this is
        a lateral approach at working height + proximity lock — the same
        mechanics the hand controller and the RL environment use.
        """
        if obj is None or self.grasped is not None:
            return False
        for _ in range(3):  # the object can shift if nudged — re-aim
            obj_pos = obj.position(self.data)
            aim = self._reachable_xy(obj_pos[:2])
            self.move_to(np.array([aim[0], aim[1], WORK_Z]), tol=0.02)
            if (
                float(np.linalg.norm(self.tip_position() - obj.position(self.data)))
                <= GRASP_DIST
            ):
                self.grasped = obj
                logger.info("Grasped %s (%s)", obj.name, obj.color)
                return True
        return False

    def release(self) -> None:
        if self.grasped is not None:
            self.data.xfrc_applied[self.grasped.body_id, :] = 0.0
            logger.info("Released %s", self.grasped.name)
        self.grasped = None
        self.step(SETTLE_STEPS)

    def place_at(self, target: np.ndarray) -> bool:
        """Carry the held object over a target point and release it there.

        The cube is released from working height and falls the last ~2 cm —
        placement is physical, consistent with the RL env's release mode.
        """
        if self.grasped is None:
            return False
        target = np.asarray(target, dtype=np.float64)
        carry_z = max(WORK_Z, float(target[2]) + 0.02)
        aim = self._reachable_xy(target[:2])
        self.move_to(np.array([aim[0], aim[1], carry_z]), tol=0.02, max_iters=16)
        placed_obj = self.grasped
        self.release()
        final = placed_obj.position(self.data)
        # PLACE_TOL is the scene's own zone-disc radius: outer targets sit at
        # the edge of the settled workspace, and landing inside the zone is
        # what the task defines as a successful placement.
        return float(np.linalg.norm(final[:2] - target[:2])) <= PLACE_TOL

    def _slot_on(self, base: SceneObject, top_half: float) -> np.ndarray:
        """World point where a carried object should rest on top of `base`."""
        return base.position(self.data) + np.array(
            [0.0, 0.0, base.half_height + top_half + 0.002]
        )

    def stack(self, top_color: str, base_color: str) -> bool:
        """Pick the top_color object and place it on the base_color object."""
        top = self.find_object(color=top_color)
        if top is None or not self.grasp(top):
            return False
        return self.stack_held_on(base_color)

    def stack_held_on(self, base_color: str) -> bool:
        """Place the currently held object onto the base_color object."""
        if self.grasped is None:
            return False
        base = self.find_object(color=base_color)
        if base is None or base is self.grasped:
            return False
        return self.place_at(self._slot_on(base, self.grasped.half_height))

    def stack_all(self) -> tuple[int, int]:
        """Build a tower from every cube in the workspace.

        Base = the cube nearest mid-workspace (keeps the growing tower
        reachable). Returns (stacked, attempted); stops at the first failure —
        physically, towers beyond 2-3 cubes exceed the carry height this arm
        can deliver, so the honest count is the result.
        """
        cubes = [
            o
            for o in self.objects
            if o.shape == "cube" and o.position(self.data)[2] > -0.5
        ]
        if len(cubes) < 2:
            return 0, 0
        mid = np.array([0.0, -0.05])
        cubes.sort(key=lambda o: float(np.linalg.norm(o.position(self.data)[:2] - mid)))
        top_of_stack = cubes[0]
        stacked = attempted = 0
        for cube in cubes[1:]:
            attempted += 1
            if not self.grasp(cube):
                break
            if not self.place_at(self._slot_on(top_of_stack, cube.half_height)):
                break
            stacked += 1
            top_of_stack = cube
        return stacked, attempted

    def resolve_location(self, name: Optional[str]) -> Optional[np.ndarray]:
        """Map a spoken location name to a world point."""
        if name in ("zone", "target"):
            p = self.place_zone_position()
            return np.array([p[0], p[1], 0.002])
        if name in ("center", "middle"):
            return np.array([0.0, -0.06, 0.002])
        if name == "corner":
            # Toward the table's near corner, at the edge of settled reach.
            d = R_REACH / np.sqrt(2.0)
            return np.array([d, -d, 0.002])
        return None

    def neutral(self) -> None:
        self.cursor = np.zeros(2, dtype=np.float64)
        self.baseline = float(NEUTRAL_LEN)
        self._cursor_slew.reset(self.cursor)
        self.step(SETTLE_STEPS)

    def nudge(self, direction: str, amount: float = 0.25) -> None:
        """Small open-loop cursor shift for MOVE commands."""
        delta = {
            "left": (-amount, 0.0, 0.0),
            "right": (amount, 0.0, 0.0),
            "forward": (0.0, amount, 0.0),
            "back": (0.0, -amount, 0.0),
            "up": (0.0, 0.0, 0.02),
            "down": (0.0, 0.0, -0.02),
        }.get(direction)
        if delta is None:
            return
        self.cursor = np.clip(self.cursor + np.array(delta[:2]), -1.0, 1.0)
        self.baseline = float(
            np.clip(self.baseline + delta[2], self.act_low.min(), self.act_high.max())
        )
        self.step(40)

    def close(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()


# ── Voice-intent execution ─────────────────────────────────────────────────────


class SimIntentExecutor:
    """Carries out validated VoiceIntents on a SimArm.

    Satisfies the IntentExecutor protocol in control/voice_commands.py.
    """

    def __init__(self, arm: SimArm):
        self.arm = arm

    @property
    def holding_object(self) -> bool:
        return self.arm.grasped is not None

    def execute(self, intent) -> str:
        from .voice_commands import CommandAction as A

        a = intent.action
        if a is A.PICK:
            obj = self.arm.find_object(color=intent.target_color)
            if obj is None:
                return "No matching object found."
            ok = self.arm.grasp(obj)
            return f"Picked up the {obj.color} {obj.shape}." if ok else (
                f"Could not reach the {obj.color} {obj.shape}."
            )
        if a is A.STACK:
            if intent.target_color is None:
                # "put it on top of the blue" — stack the held object.
                ok = self.arm.stack_held_on(intent.destination_color)
                top = "the held object"
            else:
                ok = self.arm.stack(intent.target_color, intent.destination_color)
                top = intent.target_color
            return (
                f"Stacked {top} on {intent.destination_color}."
                if ok
                else f"Stacking {top} on {intent.destination_color} failed."
            )
        if a is A.STACK_ALL:
            stacked, attempted = self.arm.stack_all()
            if attempted == 0:
                return "Not enough cubes in reach to build a tower."
            return f"Tower built: stacked {stacked} of {attempted} cubes."
        if a is A.PLACE:
            target = (
                self.arm.resolve_location(intent.location)
                if intent.location
                else self.arm.place_zone_position()
            )
            if target is None:
                return f"I don't know where '{intent.location}' is."
            ok = self.arm.place_at(target)
            where = intent.location or "the target zone"
            return f"Placed at {where}." if ok else f"Placement at {where} missed."
        if a is A.RELEASE:
            self.arm.release()
            return "Released."
        if a is A.NEUTRAL:
            self.arm.neutral()
            return "Returned to neutral."
        if a is A.MOVE:
            self.arm.nudge(intent.direction or "")
            return f"Moved {intent.direction}."
        if a is A.STOP:
            self.arm.step(5)
            return "Holding position."
        if a is A.WAVE:
            for dx in (0.5, -0.5, 0.5, -0.5, 0.0):
                self.arm.nudge("right" if dx > 0 else "left", abs(dx))
            return "Waved."
        return f"Action {a.value} is not supported in simulation."

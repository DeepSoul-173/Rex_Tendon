"""Real-time webcam hand control for the MuJoCo tentacle simulation.

Control Scheme (decoupled channels)
-----------------------------------
  Wrist horizontal position (L/R)   : Bend robot tip Left / Right
  Wrist vertical position (U/D)     : Bend robot tip Up / Down
  Twist / roll the hand             : Rotate the base (yaw) — true twist DOF,
                                      active only when the model has a base-yaw
                                      actuator; smooth/slow by design.
  Hand depth (bbox size in frame)   : Z extension — closer=extend, farther=retract
  Flat open palm (low z_variance)   : Return robot to neutral position
  Pinch THUMB + INDEX (hold ~6 fr)  : LOCK grasp on nearest object
  Pinch THUMB + MIDDLE finger       : UNLOCK / release grasped object
  SPACE key                         : UNLOCK (fallback)
  'c' key                           : Recalibrate neutral pose
  'r' key                           : Reset simulation
  'q' / ESC                         : Quit

Architecture
------------
  Two threads:
    CamThread  — continuously grabs frames + runs MediaPipe at full speed.
    Main thread — reads latest features (zero blocking), steps physics,
                  renders composite window.

Work Envelope
-------------
  |x| ≤ 0.21 m, |y| ≤ 0.30 m, z in [-0.005, 0.28] m.
  When a control action would push the tip outside, the last valid ctrl
  is restored before the next physics step.
"""

import queue
import threading
import time
import logging
from typing import Optional

import numpy as np
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
from mujoco import renderer

from ..control.geometry import convert_2d_cursor_to_target_lengths
from ..perception.scene_objects import discover_objects, find_by_color
from .smoothing import HoldToTrigger, OneEuroFilter, SlewRateLimiter
from .voice_commands import CommandAction, parse_intent
from .voice_io import start_input_thread

logger = logging.getLogger(__name__)

# ── Actuator geometry ─────────────────────────────────────────────────────────
NEUTRAL_LEN = 0.23
ACTUATOR_LO = 0.12
ACTUATOR_HI = 0.34

# ── Gesture thresholds ────────────────────────────────────────────────────────
PINCH_LOCK_THRESHOLD = 0.35
PINCH_UNLOCK_MID = 0.35
PINCH_OPEN_RATIO = 0.55
FIST_LOCK_FRAMES = 4
FLAT_ZVAR = 0.00006
FLAT_OPEN_CURL_MIN = 0.42
FLAT_PALM_FRAMES = 4

# ── Cursor mapping ────────────────────────────────────────────────────────────
# Absolute wrist-position mode: the wrist's location in the (mirrored) frame maps
# directly to the tip cursor, so moving your hand moves the robot.  'C' recenters
# the neutral hand position and sets the Z (depth) baseline.
DEAD_ZONE = 0.03  # normalized frame units around neutral that map to "no motion"
POS_GAIN = 3.2  # how far the tip swings per unit of wrist travel across the frame
DEPTH_GAIN = 4.0
# Hand-roll → base-yaw (true twist). When the model has a base-yaw actuator,
# rolling/twisting the hand rotates the whole tentacle about the vertical axis.
# This is decoupled from position control (which drives the bend), so the two
# are independent and easy to steer.
YAW_GAIN = 2.4  # base-yaw radians per radian of hand roll (clamped to joint range)
YAW_SMOOTH = 0.25  # low-pass alpha for the yaw target (stability)
DEAD_ROLL = 0.06  # hand-roll dead-zone (radians)

# ── Motion safety ─────────────────────────────────────────────────────────────
MAX_TENDON_DEV = 0.070  # max tendon offset from baseline (caps bend amount)
# Tendon slew cap in m/s. Tuned by feel, NOT derived from the old per-frame
# cap: the webcam+MediaPipe loop runs at ~15-25 fps on a laptop (not the 60 fps
# design target), and a "60 fps equivalent" rate of 1.8 m/s allowed 3x faster
# tendon motion per real frame than the old controller — felt jerky and threw
# objects. 0.8 m/s is calm at any frame rate; raise via --max-rate if sluggish.
MAX_CTRL_RATE = 0.8
STALE_CAM_TIMEOUT = 0.5  # s without a fresh camera sample → hold position

# ── Cursor filtering (one-euro mode) ──────────────────────────────────────────
# Adaptive smoothing: calm when the hand is still, near-direct when it moves
# fast. min_cutoff (Hz): lower = steadier at rest, laggier on slow drift.
# beta: higher = less lag during fast motion. "ema" mode keeps the legacy
# fixed-alpha filter as the A/B baseline.
CURSOR_MIN_CUTOFF = 1.0
CURSOR_BETA = 0.15
YAW_MIN_CUTOFF = 1.0  # yaw is deliberately slower than the bend cursor
YAW_BETA = 0.2

# ── Gesture timing (seconds; was frame counts at the 60 fps target) ──────────
FIST_LOCK_TIME = FIST_LOCK_FRAMES / 60.0
FLAT_PALM_TIME = FLAT_PALM_FRAMES / 60.0

# ── Work envelope ─────────────────────────────────────────────────────────────
TABLE_X_MAX = 0.21
TABLE_Y_MAX = 0.30
TIP_Z_MIN = -0.005
TIP_Z_MAX = 0.28

# ── Proximity grasping ────────────────────────────────────────────────────────
AUTO_WRAP_DIST = 0.09  # must be within this to auto-wrap
AUTO_GRASP_DIST = 0.055  # pinch + within this → instant lock

# ── PiP window ────────────────────────────────────────────────────────────────
PIP_W, PIP_H = 262, 196
PIP_M = 8
PIP_BORDER = 2

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_RING_MCP = 13
_PINKY_MCP = 17


# ─────────────────────────────────────────────────────────────────────────────
# Camera / MediaPipe thread
# ─────────────────────────────────────────────────────────────────────────────


class _CamThread(threading.Thread):
    """Background thread: capture + MediaPipe → always exposes latest features."""

    def __init__(self, camera_id: int):
        super().__init__(daemon=True, name="CamThread")
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

        self.latest_frame = None  # BGR ndarray, mirrored
        self.latest_features = None  # dict | None
        self.latest_wrist_pix = None  # (px, py) | None
        self.latest_timestamp = 0.0

        # MediaPipe — tuned for continuous real-time tracking
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.35,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

    def stop(self):
        self._stop_evt.set()

    def read(self):
        """Thread-safe snapshot of (frame, features, wrist_pix)."""
        with self._lock:
            return (
                None if self.latest_frame is None else self.latest_frame.copy(),
                self.latest_features,
                self.latest_wrist_pix,
                self.latest_timestamp,
            )

    def run(self):
        cap = cv2.VideoCapture(self._camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always get latest frame

        try:
            while not self._stop_evt.is_set():
                ret, raw = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(raw, 1)  # mirror: real hand L/R = screen L/R
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = self._mp_hands.process(rgb)

                features = None
                wrist_pix = None
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    # Draw on frame copy
                    self._mp_draw.draw_landmarks(
                        frame,
                        lm,
                        list(mp.solutions.hands.HAND_CONNECTIONS),
                        self._mp_draw_styles.get_default_hand_landmarks_style(),
                        self._mp_draw_styles.get_default_hand_connections_style(),
                    )
                    features = _extract_features(lm, frame.shape)
                    wx = int(lm.landmark[_WRIST].x * w)
                    wy = int(lm.landmark[_WRIST].y * h)
                    wrist_pix = (wx, wy)

                with self._lock:
                    self.latest_frame = frame
                    self.latest_features = features
                    self.latest_wrist_pix = wrist_pix
                    self.latest_timestamp = time.perf_counter()
        finally:
            cap.release()
            self._mp_hands.close()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (runs inside CamThread)
# ─────────────────────────────────────────────────────────────────────────────


def _extract_features(hand_lm, frame_shape):
    """Extract stable hand features from a single MediaPipe hand_landmarks."""
    lm = hand_lm.landmark
    h, w = frame_shape[:2]

    wrist = lm[_WRIST]

    def _d(a, b):
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    palm_diag = _d(wrist, lm[_PINKY_MCP]) + 1e-6

    # Stable absolute wrist position (normalised 0-1, frame already mirrored)
    wrist_x = float(wrist.x)  # 0=left edge, 1=right edge
    wrist_y = float(wrist.y)  # 0=top, 1=bottom

    # Bounding box area (depth proxy)
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

    # In-plane roll angle (index_mcp → pinky_mcp)
    im = lm[_INDEX_MCP]
    pm = lm[_PINKY_MCP]
    roll_angle = float(np.arctan2(im.y - pm.y, im.x - pm.x))

    # Pinch: thumb+index (lock)
    pinch_ratio = _d(lm[_THUMB_TIP], lm[_INDEX_TIP]) / palm_diag

    # Pinch: thumb+middle (unlock)
    thumb_middle_ratio = _d(lm[_THUMB_TIP], lm[_MIDDLE_TIP]) / palm_diag

    # Palm normal X -> Bend left/right
    u = np.array(
        [
            lm[_INDEX_MCP].x - wrist.x,
            lm[_INDEX_MCP].y - wrist.y,
            lm[_INDEX_MCP].z - wrist.z,
        ]
    )
    v = np.array(
        [
            lm[_PINKY_MCP].x - wrist.x,
            lm[_PINKY_MCP].y - wrist.y,
            lm[_PINKY_MCP].z - wrist.z,
        ]
    )
    normal = np.cross(u, v)
    normal_mag = np.linalg.norm(normal)
    normal_norm = normal / normal_mag if normal_mag > 1e-6 else np.array([0, 0, 1])
    palm_normal_x = float(normal_norm[0])

    # Forward vector -> Bend up/down
    w = np.array(
        [
            lm[_MIDDLE_MCP].x - wrist.x,
            lm[_MIDDLE_MCP].y - wrist.y,
            lm[_MIDDLE_MCP].z - wrist.z,
        ]
    )
    w_mag = np.linalg.norm(w)
    forward_norm = w / w_mag if w_mag > 1e-6 else np.array([0, 1, 0])
    forward_y = float(forward_norm[1])

    target_yaw = float(np.arcsin(np.clip(palm_normal_x, -1.0, 1.0)))
    target_pitch = float(np.arcsin(np.clip(forward_y, -1.0, 1.0)))

    # Flat palm: z-variance of all five fingertips  (low = flat)
    z_tips = [
        lm[_THUMB_TIP].z,
        lm[_INDEX_TIP].z,
        lm[_MIDDLE_TIP].z,
        lm[_RING_TIP].z,
        lm[_PINKY_TIP].z,
    ]
    z_variance = float(np.var(z_tips))

    # Finger curl: how curled are the four fingers?
    # (used to confirm flat-palm / fist states)
    finger_tips = [lm[_INDEX_TIP], lm[_MIDDLE_TIP], lm[_RING_TIP], lm[_PINKY_TIP]]
    finger_mcps = [lm[_INDEX_MCP], lm[_MIDDLE_MCP], lm[_RING_MCP], lm[_PINKY_MCP]]
    avg_curl = float(np.mean([_d(t, m) for t, m in zip(finger_tips, finger_mcps)]))

    return {
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
        "bbox_area": bbox_area,
        "roll_angle": roll_angle,
        "pinch_ratio": float(pinch_ratio),
        "thumb_middle_ratio": float(thumb_middle_ratio),
        "palm_normal_x": palm_normal_x,
        "target_yaw": target_yaw,
        "target_pitch": target_pitch,
        "z_variance": z_variance,
        "avg_curl": avg_curl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cursor mapping helpers
# ─────────────────────────────────────────────────────────────────────────────


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _map_cursor(
    features: dict, bbox_center: float, neutral_xy: np.ndarray
) -> tuple:
    """Convert the wrist's frame position to a robot cursor + depth baseline.

    Pure position control, decoupled from twist (which drives the base yaw):
      wrist_x (0=left, 1=right; frame is mirrored)  →  cursor_x in [-1, +1]
      wrist_y (0=top,  1=bottom)                    →  cursor_y in [+1, -1]  (up=+)

    'neutral_xy' is the calibrated hand centre ('C' key; default frame centre).

    Returns:
        cursor   : np.ndarray [x, y] in [-1, 1]
        baseline : float tendon length in [ACTUATOR_LO, ACTUATOR_HI]
    """
    dx = float(features["wrist_x"]) - float(neutral_xy[0])  # + = hand moved right
    dy = -(float(features["wrist_y"]) - float(neutral_xy[1]))  # + = hand moved up
    dist = float(np.hypot(dx, dy))
    if dist < DEAD_ZONE:
        cursor = np.zeros(2, dtype=np.float32)
    else:
        cx = dx * POS_GAIN
        cy = dy * POS_GAIN
        mag = float(np.hypot(cx, cy))
        if mag > 1.0:
            cx, cy = cx / mag, cy / mag
        cursor = np.array([cx, cy], dtype=np.float32)

    # Depth: bbox_area change from calibrated baseline.
    t = np.clip(0.5 + (features["bbox_area"] - bbox_center) * DEPTH_GAIN, 0.0, 1.0)
    baseline = float(ACTUATOR_LO + t * (ACTUATOR_HI - ACTUATOR_LO))
    return cursor, baseline


# ─────────────────────────────────────────────────────────────────────────────
# HUD drawing
# ─────────────────────────────────────────────────────────────────────────────


def _tip_in_envelope(p) -> bool:
    return (
        abs(p[0]) <= TABLE_X_MAX
        and abs(p[1]) <= TABLE_Y_MAX
        and TIP_Z_MIN <= p[2] <= TIP_Z_MAX
    )


def _draw_hud(frame, cursor, feat, grasp_locked, tip_pos, ctrl, mode, status, fist_cnt):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 126), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    oob = not _tip_in_envelope(tip_pos)
    locked = grasp_locked
    gcol = (0, 165, 255) if locked else (0, 210, 80)  # orange lock / green free
    gtxt = "CARRYING (move hand to steer!)" if locked else "FREE"
    charge = f"{min(fist_cnt, FIST_LOCK_FRAMES)}/{FIST_LOCK_FRAMES}"

    if feat:
        pr = f"{feat['pinch_ratio']:.2f}"
        mr = f"{feat['thumb_middle_ratio']:.2f}"
        wx = f"{feat['wrist_x']:.2f}"
        wy = f"{feat['wrist_y']:.2f}"
        curl = f"{feat['avg_curl']:.3f}"
    else:
        pr = mr = wx = wy = curl = "---"

    lines = [
        (f"MODE:{mode.upper()}  GRASP:{gtxt}  Charge:{charge}", gcol),
        (
            f"Wrist X:{wx} Y:{wy}   Cursor({cursor[0]:+.2f},{cursor[1]:+.2f})",
            (200, 200, 200),
        ),
        (f"Pinch(lock):{pr}   Mid(unlock):{mr}   Curl:{curl}", (200, 200, 200)),
        (
            f"Tip ({tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f})",
            (0, 40, 255) if oob else (160, 210, 255),
        ),
        (f"Ctrl ({ctrl[0]:.3f}, {ctrl[1]:.3f}, {ctrl[2]:.3f})", (200, 200, 200)),
        (status, (255, 200, 50)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (10, 20 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            col,
            1,
            cv2.LINE_AA,
        )

    # Large banner when actively carrying an object
    if locked:
        bh = 32
        cv2.rectangle(
            frame, (0, h - bh - 12), (w, h - 12), (0, 100, 200), -1
        )  # blue-orange bar
        cv2.putText(
            frame,
            "CARRYING 🔒  Move hand to steer | Thumb+MIDDLE to release",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    elif oob:
        cv2.putText(
            frame,
            "OUT OF WORK ENVELOPE",
            (w // 2 - 160, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        legend = (
            "SPACE=release  C=calib-depth  R=reset  Q/ESC=quit  "
            "| THUMB+INDEX=grasp   THUMB+MIDDLE=release"
        )
        cv2.putText(
            frame,
            legend,
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main controller
# ─────────────────────────────────────────────────────────────────────────────


class HandSimController:
    """Threaded, wrist-position hand-controlled MuJoCo tentacle simulation."""

    def __init__(
        self,
        xml_path: str,
        model_path: Optional[str] = None,
        camera_id: int = 0,
        mode: str = "direct",
        smoothing_alpha: float = 0.90,
        filter_mode: str = "one-euro",
        min_cutoff: float = CURSOR_MIN_CUTOFF,
        beta: float = CURSOR_BETA,
        max_ctrl_rate: float = MAX_CTRL_RATE,
        voice: Optional[str] = None,  # None | "typed" | "speech"
    ):
        if filter_mode not in ("one-euro", "ema"):
            raise ValueError(
                f"filter_mode must be 'one-euro' or 'ema', got {filter_mode!r}"
            )
        self.mode = mode
        self.alpha = smoothing_alpha
        self.filter_mode = filter_mode
        self._filter_desc = (
            f"1euro mc={min_cutoff:g} b={beta:g}"
            if filter_mode == "one-euro"
            else f"ema a={smoothing_alpha:g}"
        )

        # MuJoCo
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self._rend_robot = renderer.Renderer(
            self.mj_model, width=PIP_W * 2, height=PIP_H * 2
        )
        self._rend_corner = renderer.Renderer(
            self.mj_model, width=PIP_W * 2, height=PIP_H * 2
        )

        # Tendon actuators are the first 3. An optional 4th actuator is the
        # base yaw (true twist); the controller adapts to its presence.
        self.act_low = self.mj_model.actuator_ctrlrange[:3, 0]
        self.act_high = self.mj_model.actuator_ctrlrange[:3, 1]
        self._has_yaw = int(self.mj_model.nu) >= 4
        if self._has_yaw:
            self._yaw_lo = float(self.mj_model.actuator_ctrlrange[3, 0])
            self._yaw_hi = float(self.mj_model.actuator_ctrlrange[3, 1])
        else:
            self._yaw_lo, self._yaw_hi = -1.5708, 1.5708
        self._smoothed_yaw = 0.0

        # Control state
        self.current_ctrl = np.full(3, NEUTRAL_LEN, dtype=np.float32)
        self._last_valid_ctrl = self.current_ctrl.copy()
        self.smoothed_cursor = np.zeros(2, dtype=np.float32)
        self.smoothed_baseline = NEUTRAL_LEN
        self._write_ctrl()

        # dt-aware signal conditioning (see smoothing.py)
        self._cursor_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._baseline_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._yaw_filter = OneEuroFilter(min_cutoff=YAW_MIN_CUTOFF, beta=YAW_BETA)
        self._ctrl_slew = SlewRateLimiter(max_ctrl_rate)
        self._ctrl_slew.reset(self.current_ctrl)

        self.grasp_locked = False
        self.grasped_bid = -1
        self._pinch_hold = HoldToTrigger(FIST_LOCK_TIME)
        self._flat_hold = HoldToTrigger(FLAT_PALM_TIME)
        self._pinch_was_open = True
        self._grasp_offset = np.zeros(3, dtype=np.float32)

        # Depth calibration only (wrist position does not need calibration)
        self._bbox_center = 0.05  # updated on 'C'
        self._neutral_xy = np.array([0.5, 0.5], dtype=np.float32)
        self._neutral_roll: Optional[float] = None  # captured on first hand / 'C'

        # Site / object IDs
        self.tip_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "tip_center"
        )
        self.ghost_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "ghost_cursor"
        )

        # Graspable objects discovered from the loaded scene (any obj_* body),
        # so swapping scenes never needs a code change.
        self.scene_objects = discover_objects(self.mj_model)
        self.object_body_ids = [o.body_id for o in self.scene_objects]

        # RL-guided mode is NOT implemented: the loop below is pure direct
        # teleoperation and never queried the loaded policy (it expects the
        # training env's stacked observation vector, which this sim does not
        # build). Be honest instead of silently behaving like direct mode.
        self.rl_model = None
        if mode == "rl":
            logger.warning(
                "--mode rl is not implemented in the hand controller; "
                "falling back to DIRECT control. (Wiring the policy needs the "
                "training env's observation pipeline — future work.)"
            )
            self.mode = "direct"

        # Camera presence
        def _cam_exists(name):
            return (
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
            )

        self._has_robot_cam = _cam_exists("robot_cam")
        self._has_corner_cam = _cam_exists("table_corner_cam")

        # Voice co-pilot: hand steers continuous motion, voice handles the
        # discrete decisions (grab <color> / release / neutral / stop).
        self._voice_queue: Optional[queue.Queue] = None
        self._voice_target = None  # SceneObject armed for voice-assisted grasp
        self._voice_status = ""
        if voice in ("typed", "speech"):
            self._voice_queue = queue.Queue()
            self._voice_status = "voice co-pilot starting..."
            # Mic ON/OFF status lands on the HUD via _voice_status.
            start_input_thread(
                self._voice_queue,
                mode=voice,
                status_cb=lambda msg: setattr(self, "_voice_status", msg),
            )

        self.camera_id = camera_id
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _write_ctrl(self) -> None:
        """Write tendon ctrl (and base-yaw ctrl if the model has that DOF)."""
        self.mj_data.ctrl[:3] = self.current_ctrl
        if self._has_yaw:
            self.mj_data.ctrl[3] = self._smoothed_yaw

    def _reset_motion_filters(self) -> None:
        """Clear filter state so motion resumes cleanly from current_ctrl."""
        self._cursor_filter.reset()
        self._baseline_filter.reset()
        self._yaw_filter.reset()
        self._ctrl_slew.reset(self.current_ctrl)

    # ── Grasp helpers ──────────────────────────────────────────────────────────

    def _nearest_object(self):
        tip = self.mj_data.site_xpos[self.tip_site_id].copy()
        best_d, best_bid = float("inf"), -1
        for bid in self.object_body_ids:
            d = float(np.linalg.norm(tip - self.mj_data.xpos[bid]))
            if d < best_d:
                best_d, best_bid = d, bid
        return best_bid, best_d

    def _try_lock_grasp(self, preferred_bid: int = -1):
        if self.grasp_locked:
            return
        self.grasp_locked = True
        self._pinch_hold.reset()
        self._pinch_was_open = False

        bid, dist = self._nearest_object()
        tip_pos = self.mj_data.site_xpos[self.tip_site_id].copy()
        if preferred_bid >= 0:
            p_dist = float(
                np.linalg.norm(tip_pos - self.mj_data.xpos[preferred_bid])
            )
            if p_dist <= AUTO_WRAP_DIST:
                bid, dist = preferred_bid, p_dist

        if dist <= AUTO_WRAP_DIST and bid >= 0:
            self.grasped_bid = bid

            # Snap object neatly beneath the tip to avoid collision explosion.
            # A fixed offset ensures it doesn't float far away if grabbed from a distance.
            jnt_adr = self.mj_model.body_jntadr[bid]
            if jnt_adr >= 0:
                qvel_adr = self.mj_model.jnt_dofadr[jnt_adr]

                # Hang the object 2.2 cm below the tip: clear of the tip's
                # 8 mm contact sphere, else the per-frame snap re-creates
                # penetration and the contact impulses jitter the arm.
                self._grasp_offset = np.array([0.0, 0.0, -0.022], dtype=np.float32)

                # Freeze the object at this exact offset
                if qvel_adr >= 0:
                    self.mj_data.qvel[qvel_adr : qvel_adr + 6] = 0.0

            # No pre-bend: the old "wrap pose" overwrote current_ctrl in a
            # single frame, bypassing the slew limiter — a violent jerk on
            # every grasp. The hand keeps continuous control; the snap-carry
            # alone provides the hold.

            logger.info(f"Grasp LOCKED + auto-wrap → dist={dist:.3f} m")
        else:
            self.grasped_bid = -1
            logger.info(f"Grasp LOCKED (frozen posture). Nearest={dist:.3f} m")

    def _unlock_grasp(self):
        if not self.grasp_locked:
            return
        self.grasp_locked = False
        self.grasped_bid = -1
        self._pinch_hold.reset()
        self._grasp_offset = np.zeros(3, dtype=np.float32)
        logger.info("Grasp UNLOCKED.")

    # ── Voice co-pilot ─────────────────────────────────────────────────────────

    def _drain_voice_commands(self) -> None:
        """Apply queued voice commands — discrete decisions only; the hand
        keeps continuous control of motion."""
        if self._voice_queue is None:
            return
        while True:
            try:
                text = self._voice_queue.get_nowait()
            except queue.Empty:
                return
            intent = parse_intent(text)
            if intent is None:
                self._voice_status = f"voice: didn't get '{text.strip()}'"
                continue
            a = intent.action
            if a is CommandAction.PICK:
                obj = (
                    find_by_color(self.scene_objects, intent.target_color)
                    if intent.target_color
                    else "nearest"
                )
                if obj is None:
                    self._voice_status = (
                        f"voice: no {intent.target_color} object in this scene"
                    )
                    continue
                self._voice_target = obj
                name = (
                    "nearest object"
                    if obj == "nearest"
                    else f"{obj.color} {obj.shape}"
                )
                self._voice_status = f"voice: GRAB {name} — steer close"
            elif a is CommandAction.RELEASE:
                self._unlock_grasp()
                self._voice_target = None
                self._voice_status = "voice: released"
            elif a is CommandAction.NEUTRAL:
                if not self.grasp_locked:
                    self.smoothed_cursor = np.zeros(2, dtype=np.float32)
                    self.smoothed_baseline = NEUTRAL_LEN
                    self.current_ctrl = np.full(3, NEUTRAL_LEN, dtype=np.float32)
                    self._reset_motion_filters()
                self._voice_status = "voice: neutral"
            elif a is CommandAction.STOP:
                self._voice_target = None
                self._voice_status = "voice: standby"
            else:
                self._voice_status = (
                    f"voice: '{a.value}' is autonomous — use run_voice_sim.py"
                )

    def _voice_assisted_grasp(self) -> Optional[str]:
        """Lock the armed voice target once the hand steers the tip in range."""
        if self._voice_target is None or self.grasp_locked:
            return None
        tip = self.mj_data.site_xpos[self.tip_site_id]
        if self._voice_target == "nearest":
            _, dist = self._nearest_object()
            if dist <= AUTO_WRAP_DIST:
                self._try_lock_grasp()
                self._voice_target = None
                self._voice_status = "VOICE GRASP locked"
                return self._voice_status
            return None
        d = float(
            np.linalg.norm(tip - self.mj_data.xpos[self._voice_target.body_id])
        )
        if d <= AUTO_WRAP_DIST:
            msg = (
                f"VOICE GRASP: {self._voice_target.color} "
                f"{self._voice_target.shape}"
            )
            self._try_lock_grasp(preferred_bid=self._voice_target.body_id)
            self._voice_target = None
            self._voice_status = msg
            return msg
        return None

    # ── Work-envelope enforcement ──────────────────────────────────────────────

    def _enforce_envelope(self):
        tip = self.mj_data.site_xpos[self.tip_site_id]
        if _tip_in_envelope(tip):
            self._last_valid_ctrl = self.mj_data.ctrl[:3].copy()
        else:
            self.mj_data.ctrl[:3] = self._last_valid_ctrl
            self.current_ctrl = self._last_valid_ctrl.copy()

    # ── Object teleport (called once per frame, AFTER all mj_steps) ───────────

    def _hold_grasped_object(self):
        """Snap the grasped object to the robot tip every frame.

        The object is placed exactly at the tip-center site so it visually
        sits inside the coiled robot.  Velocities are zeroed to stop drift.
        mj_forward() is called once so xpos/xmat reflect the new position
        before viewer.sync().
        """
        if not self.grasp_locked or self.grasped_bid < 0:
            return
        bid = self.grasped_bid
        jnt_adr = self.mj_model.body_jntadr[bid]
        if jnt_adr < 0:
            return
        qpos_adr = self.mj_model.jnt_qposadr[jnt_adr]
        qvel_adr = self.mj_model.jnt_dofadr[jnt_adr]

        # Snap object to tip maintaining initial offset
        tip_pos = self.mj_data.site_xpos[self.tip_site_id].copy()
        self.mj_data.qpos[qpos_adr : qpos_adr + 3] = tip_pos + self._grasp_offset
        # Zero all 6 DOF velocities (3 linear + 3 angular) to prevent drift
        if qvel_adr >= 0:
            self.mj_data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        # Propagate so the renderer sees the correct object position
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        cam_thread = _CamThread(self.camera_id)
        cam_thread.start()

        # Give camera a moment to initialise
        time.sleep(0.3)

        viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        WIN = "Rex Tendon — Hand Control"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        FRAME_SKIP = 6  # lower latency physics batches
        FRAME_TIME = 1.0 / 60.0  # target 60 fps display/control
        status_msg = (
            "Press C to calibrate neutral hand pose  |  "
            "THUMB+INDEX=lock  |  THUMB+MIDDLE=unlock"
        )

        prev_tick: Optional[float] = None

        try:
            while viewer.is_running():
                t0 = time.time()

                # Measured dt feeds the dt-aware filters. Clamped so the first
                # frame and long stalls can't blow up the velocity estimate.
                now = time.perf_counter()
                dt = FRAME_TIME if prev_tick is None else now - prev_tick
                prev_tick = now
                dt = min(max(dt, 1e-3), 0.1)

                # ── Non-blocking camera read ───────────────────────────────────
                frame, features, wrist_pix, cam_ts = cam_thread.read()
                if frame is None:
                    # No frame yet; show blank and skip
                    time.sleep(0.01)
                    continue

                # ── Voice co-pilot: discrete commands + assisted grasp ────────
                self._drain_voice_commands()
                voice_msg = self._voice_assisted_grasp()
                if voice_msg:
                    status_msg = voice_msg

                # ── Real-Time Gesture & Fluid Control ─────────────────────────
                # Freshness gate: the camera mailbox keeps the last sample
                # forever, so a stalled cam thread would otherwise keep
                # commanding the robot with frozen features.
                cam_stale = (
                    cam_ts <= 0.0
                    or (time.perf_counter() - cam_ts) > STALE_CAM_TIMEOUT
                )

                if cam_stale:
                    status_msg = "CAMERA STALE — holding position"
                elif features is not None:
                    pinch = features["pinch_ratio"]
                    mid = features["thumb_middle_ratio"]
                    curl = features["avg_curl"]
                    zvar = features["z_variance"]

                    # Capture the neutral hand-roll once so twist is measured
                    # relative to however the hand is first held.
                    if self._neutral_roll is None:
                        self._neutral_roll = float(features["roll_angle"])

                    # 1. Flat-palm override → go neutral (held FLAT_PALM_TIME s)
                    self._flat_hold.update(
                        zvar < FLAT_ZVAR and curl > FLAT_OPEN_CURL_MIN, dt
                    )

                    flat_triggered = False
                    if self._flat_hold.progress >= 1.0:
                        if not self.grasp_locked:
                            self.smoothed_cursor = np.zeros(2, dtype=np.float32)
                            self.smoothed_baseline = NEUTRAL_LEN
                            self.current_ctrl = np.full(
                                3, NEUTRAL_LEN, dtype=np.float32
                            )
                            self._reset_motion_filters()
                            flat_triggered = True
                        status_msg = "FLAT PALM → NEUTRAL"

                    if not flat_triggered:
                        # 2. Unlock gesture: thumb + middle
                        if self.grasp_locked and mid < PINCH_UNLOCK_MID:
                            self._unlock_grasp()
                            status_msg = "THUMB+MIDDLE → RELEASED 🔓"

                        # 3. Lock gesture: thumb + index
                        elif not self.grasp_locked:
                            if pinch > PINCH_OPEN_RATIO:
                                self._pinch_was_open = True

                            pinch_active = (
                                pinch < PINCH_LOCK_THRESHOLD and self._pinch_was_open
                            )
                            held_long_enough = self._pinch_hold.update(
                                pinch_active, dt
                            )
                            if pinch_active:
                                _, near_d = self._nearest_object()
                                if near_d <= AUTO_GRASP_DIST:
                                    self._try_lock_grasp()
                                    status_msg = f"AUTO-GRASP (dist {near_d:.3f} m) 🔒"
                                elif held_long_enough:
                                    self._try_lock_grasp()
                                    status_msg = "PINCH HELD → LOCKED 🔒"
                                else:
                                    pct = int(100 * self._pinch_hold.progress)
                                    status_msg = f"Charging grasp… {pct}%"

                        # 4. Continuous motion — "Move like water"
                        # Position drives the bend; hand-roll drives the base yaw.
                        cursor, baseline = _map_cursor(
                            features, self._bbox_center, self._neutral_xy
                        )

                        # Hand twist → base yaw (decoupled true-twist DOF).
                        if self._has_yaw and self._neutral_roll is not None:
                            roll_delta = _wrap_angle(
                                float(features["roll_angle"]) - self._neutral_roll
                            )
                            if abs(roll_delta) < DEAD_ROLL:
                                roll_delta = 0.0
                            yaw_target = float(
                                np.clip(
                                    -roll_delta * YAW_GAIN, self._yaw_lo, self._yaw_hi
                                )
                            )
                            if self.filter_mode == "one-euro":
                                self._smoothed_yaw = float(
                                    self._yaw_filter.filter(yaw_target, dt)
                                )
                            else:
                                self._smoothed_yaw = (
                                    YAW_SMOOTH * yaw_target
                                    + (1.0 - YAW_SMOOTH) * self._smoothed_yaw
                                )

                        if self.filter_mode == "one-euro":
                            # Adaptive: calm when the hand is still (jitter
                            # gone), near-direct when it moves fast (no lag).
                            self.smoothed_cursor = np.asarray(
                                self._cursor_filter.filter(cursor, dt),
                                dtype=np.float32,
                            )
                            self.smoothed_baseline = float(
                                self._baseline_filter.filter(baseline, dt)
                            )
                        else:
                            # Legacy fixed-alpha EMA, kept as the A/B baseline.
                            # High alpha means "follow me now"; the slew limiter
                            # provides the remaining physical smoothing.
                            eff_alpha = self.alpha
                            self.smoothed_cursor = (
                                eff_alpha * cursor
                                + (1.0 - eff_alpha) * self.smoothed_cursor
                            )
                            self.smoothed_baseline = (
                                eff_alpha * baseline
                                + (1.0 - eff_alpha) * self.smoothed_baseline
                            )

                        # Ghost cursor (hide when locked)
                        if self.ghost_site_id >= 0:
                            if self.grasp_locked:
                                self.mj_model.site_rgba[self.ghost_site_id] = [
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                            else:
                                g3d = np.array(
                                    [
                                        self.smoothed_cursor[0] * 0.2,
                                        self.smoothed_cursor[1] * 0.2,
                                        (self.smoothed_baseline - NEUTRAL_LEN) * 0.5
                                        + 0.05,
                                    ],
                                    dtype=np.float32,
                                )
                                self.mj_model.site_pos[self.ghost_site_id] = g3d
                                gcol = (
                                    [0, 1, 0, 0.7]
                                    if pinch < PINCH_LOCK_THRESHOLD
                                    else [1, 1, 1, 0.3]
                                )
                                self.mj_model.site_rgba[self.ghost_site_id] = gcol

                        bl_arr = np.full(3, self.smoothed_baseline, dtype=np.float32)
                        target_ctrl = convert_2d_cursor_to_target_lengths(
                            self.smoothed_cursor,
                            bl_arr,
                            self.act_low,
                            self.act_high,
                            1.0,
                        )

                        # Deviation cap
                        deviation = target_ctrl - self.smoothed_baseline
                        deviation = np.clip(deviation, -MAX_TENDON_DEV, MAX_TENDON_DEV)
                        target_ctrl = self.smoothed_baseline + deviation
                        target_ctrl = np.clip(target_ctrl, self.act_low, self.act_high)

                        # Slew limiter: dt-aware tendon rate cap (units/second;
                        # equals the old 0.03/frame cap at the 60 fps target).
                        self.current_ctrl = np.asarray(
                            self._ctrl_slew.step(target_ctrl, dt),
                            dtype=np.float32,
                        )

                        grip_st = "CARRYING 🔒" if self.grasp_locked else "FREE"
                        status_msg = f"Wx:{features['wrist_x']:.2f}  Wy:{features['wrist_y']:.2f}  [{grip_st}]"
                else:
                    status_msg = "No hand detected — holding state"

                # ── Physics step ───────────────────────────────────────────────
                self._write_ctrl()

                # Pre-compute anti-gravity force for grasped object once.
                # xfrc_applied[bid] = [fx, fy, fz, tx, ty, tz] in world frame.
                if self.grasp_locked and self.grasped_bid >= 0:
                    g_bid = self.grasped_bid
                    g_mass = float(self.mj_model.body_mass[g_bid])
                    # Full anti-gravity (counteracts 9.81 downward)
                    self.mj_data.xfrc_applied[g_bid, :] = [
                        0.0,
                        0.0,
                        g_mass * 9.81,
                        0.0,
                        0.0,
                        0.0,
                    ]
                else:
                    # Clear any residual force on all objects
                    for _bid in self.object_body_ids:
                        self.mj_data.xfrc_applied[_bid, :] = 0.0

                for _ in range(FRAME_SKIP):
                    # PRE-STEP FREEZE: snap object to tip before physics runs.
                    # This prevents gravity / contacts from accumulating between snaps.
                    if self.grasp_locked and self.grasped_bid >= 0:
                        g_jnt = self.mj_model.body_jntadr[self.grasped_bid]
                        if g_jnt >= 0:
                            gq = self.mj_model.jnt_qposadr[g_jnt]
                            gv = self.mj_model.jnt_dofadr[g_jnt]
                            t_pos = self.mj_data.site_xpos[self.tip_site_id].copy()
                            self.mj_data.qpos[gq : gq + 3] = t_pos + self._grasp_offset
                            if gv >= 0:
                                self.mj_data.qvel[gv : gv + 6] = 0.0
                    mujoco.mj_step(self.mj_model, self.mj_data)

                # Post-loop: final accurate snap + forward kinematics
                self._hold_grasped_object()

                # Enforce work envelope AFTER teleport
                self._enforce_envelope()

                viewer.sync()

                # ── Draw HUD on webcam frame ───────────────────────────────────
                tip_pos = self.mj_data.site_xpos[self.tip_site_id]
                _draw_hud(
                    frame,
                    self.smoothed_cursor,
                    features,
                    self.grasp_locked,
                    tip_pos,
                    self.current_ctrl,
                    f"{self.mode} [{self._filter_desc}]",
                    status_msg,
                    int(round(self._pinch_hold.progress * FIST_LOCK_FRAMES)),
                )

                # Voice co-pilot status line (below the HUD block)
                if self._voice_queue is not None and self._voice_status:
                    cv2.putText(
                        frame,
                        self._voice_status,
                        (10, 142),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (80, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Movement arrow from wrist
                # GREEN = free tracking  |  ORANGE = carrying (still steerable)
                if wrist_pix is not None:
                    cx, cy = wrist_pix
                    cur = self.smoothed_cursor
                    ARROW = 75
                    ax = int(cx + cur[0] * ARROW)
                    ay = int(cy - cur[1] * ARROW)
                    col = (0, 165, 255) if self.grasp_locked else (0, 230, 90)
                    if np.linalg.norm(cur) > 0.04:
                        cv2.arrowedLine(
                            frame, (cx, cy), (ax, ay), col, 3, tipLength=0.3
                        )
                    cv2.circle(frame, (cx, cy), 12, col, 2)
                    cv2.circle(frame, (cx, cy), 4, col, -1)
                    label = "CARRY" if self.grasp_locked else "STEER"
                    cv2.putText(
                        frame,
                        label,
                        (cx + 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

                # ── PiP overlays ───────────────────────────────────────────────
                mh, mw = frame.shape[:2]

                if self._has_robot_cam:
                    try:
                        self._rend_robot.update_scene(self.mj_data, camera="robot_cam")
                        rc = cv2.cvtColor(self._rend_robot.render(), cv2.COLOR_RGB2BGR)
                        _pip(
                            frame,
                            rc,
                            mw - PIP_W - PIP_M,
                            PIP_M,
                            PIP_W,
                            PIP_H,
                            "Robot Cam",
                        )
                    except Exception as e:
                        logger.debug(f"robot_cam: {e}")

                if self._has_corner_cam:
                    try:
                        self._rend_corner.update_scene(
                            self.mj_data, camera="table_corner_cam"
                        )
                        cc = cv2.cvtColor(self._rend_corner.render(), cv2.COLOR_RGB2BGR)
                        py2 = PIP_M + PIP_H + PIP_M + 2 * PIP_BORDER
                        _pip(
                            frame,
                            cc,
                            mw - PIP_W - PIP_M,
                            py2,
                            PIP_W,
                            PIP_H,
                            "Table Corner Cam",
                        )
                    except Exception as e:
                        logger.debug(f"corner_cam: {e}")

                cv2.imshow(WIN, frame)

                # ── Keyboard ───────────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    self._unlock_grasp()
                    status_msg = "SPACE → RELEASED 🔓"
                elif key == ord("c"):
                    # Calibrate neutral hand position and depth baseline.
                    if features is not None:
                        self._bbox_center = features["bbox_area"]
                        self._neutral_xy = np.array(
                            [features["wrist_x"], features["wrist_y"]],
                            dtype=np.float32,
                        )
                        self._neutral_roll = float(features["roll_angle"])
                    self.smoothed_cursor = np.zeros(2, dtype=np.float32)
                    self.smoothed_baseline = NEUTRAL_LEN
                    self._reset_motion_filters()
                    status_msg = "Neutral hand pose calibrated"
                elif key == ord("r"):
                    mujoco.mj_resetData(self.mj_model, self.mj_data)
                    self.current_ctrl = np.full(3, NEUTRAL_LEN, dtype=np.float32)
                    self._last_valid_ctrl = self.current_ctrl.copy()
                    self._smoothed_yaw = 0.0
                    self._write_ctrl()
                    self.grasp_locked = False
                    self.grasped_bid = -1
                    self._pinch_hold.reset()
                    self._flat_hold.reset()
                    self._pinch_was_open = True
                    self._grasp_offset = np.zeros(3, dtype=np.float32)
                    self.smoothed_cursor = np.zeros(2, dtype=np.float32)
                    self.smoothed_baseline = NEUTRAL_LEN
                    self._reset_motion_filters()
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    status_msg = "↺ RESET"

                # Frame-rate cap
                elapsed = time.time() - t0
                remaining = FRAME_TIME - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cam_thread.stop()
            cam_thread.join(timeout=2.0)
            cv2.destroyAllWindows()
            if viewer.is_running():
                viewer.close()
            self._rend_robot.close()
            self._rend_corner.close()
            logger.info("Hand controller shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# PiP helper
# ─────────────────────────────────────────────────────────────────────────────


def _pip(dst, src_bgr, x, y, w, h, label=""):
    img = cv2.resize(src_bgr, (w, h))
    cv2.rectangle(
        dst,
        (x - PIP_BORDER, y - PIP_BORDER),
        (x + w + PIP_BORDER, y + h + PIP_BORDER),
        (60, 60, 60),
        PIP_BORDER,
    )
    dst[y : y + h, x : x + w] = img
    if label:
        cv2.rectangle(dst, (x, y), (x + w, y + 18), (0, 0, 0), -1)
        cv2.putText(
            dst,
            label,
            (x + 4, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_hand_controller(
    xml_path: str = "rex_assets/rex_simulation/pick_and_place_scene.xml",
    model_path: Optional[str] = None,
    camera_id: int = 0,
    mode: str = "direct",
    smoothing: float = 0.90,  # ema mode only: high alpha = direct/physical feel
    filter_mode: str = "one-euro",
    min_cutoff: float = CURSOR_MIN_CUTOFF,
    beta: float = CURSOR_BETA,
    max_ctrl_rate: float = MAX_CTRL_RATE,
    voice: Optional[str] = None,
):
    """Launch the threaded hand controller."""
    HandSimController(
        xml_path=xml_path,
        model_path=model_path,
        camera_id=camera_id,
        mode=mode,
        smoothing_alpha=smoothing,
        filter_mode=filter_mode,
        min_cutoff=min_cutoff,
        beta=beta,
        max_ctrl_rate=max_ctrl_rate,
        voice=voice,
    ).run()

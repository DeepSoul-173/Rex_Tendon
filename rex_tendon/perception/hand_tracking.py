"""Hand tracking using MediaPipe for gesture recognition and 3D positioning."""

import threading
from typing import Optional, Tuple, Any, Deque
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import logging

logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Thread-local storage for MediaPipe instances
_thread_local = threading.local()

# Hand landmark indices
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
WRIST = 0
INDEX_FINGER_MCP = 5
MIDDLE_FINGER_MCP = 9
PINKY_MCP = 17


def _get_thread_hands_detector():
    """Get thread-local MediaPipe hands detector."""
    if (
        not hasattr(_thread_local, "hands_detector")
        or _thread_local.hands_detector is None
    ):
        _thread_local.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.25,
        )
    return _thread_local.hands_detector


def get_mediapipe_hand_data(
    frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """Detect index finger tip using MediaPipe (convenience function).

    Args:
        frame: Input image as BGR numpy array

    Returns:
        Tuple of (index_finger_tip_position, mediapipe_results)
    """
    hands_detector = _get_thread_hands_detector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)

    index_finger_tip_pos = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        tip_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
        h, w = frame.shape[:2]
        index_finger_tip_pos = np.array(
            [tip_landmark.x * w, tip_landmark.y * h], dtype=np.float32
        )

    return index_finger_tip_pos, results


def draw_hand_landmarks(
    image: np.ndarray,
    results: Any,
    connections_color: Tuple[int, int, int] = (0, 255, 0),
    landmarks_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw hand landmarks and connections on image.

    Args:
        image: Input image to draw on
        results: MediaPipe results object
        connections_color: Color for hand connections (BGR)
        landmarks_color: Color for landmarks (BGR)

    Returns:
        Image with drawn landmarks
    """
    annotated_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    return annotated_image


def is_wave_gesture(
    x_coordinates: deque,
    amplitude_threshold: float = 0.2,
    min_coordinates: int = 10,
    min_velocity_points: int = 10,
    min_zero_crossings: int = 5,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect wave gesture from x-coordinate trail.

    Args:
        x_coordinates: Deque of x-coordinate values
        amplitude_threshold: Minimum amplitude for wave detection
        min_coordinates: Minimum number of coordinate points needed
        min_velocity_points: Minimum velocity points for analysis
        min_zero_crossings: Minimum zero crossings for wave detection

    Returns:
        Tuple of (is_wave_detected, velocity_array)
    """
    if len(x_coordinates) < min_coordinates:
        return False, None

    x_array = np.array(list(x_coordinates))
    velocity = np.diff(x_array)

    if len(velocity) < min_velocity_points:
        return False, None

    # Find zero crossings in velocity (direction changes)
    zero_crossings_indices = np.where(
        np.sign(velocity[:-1]) * np.sign(velocity[1:]) < 0
    )[0]
    num_zero_crossings = len(zero_crossings_indices)

    # Calculate amplitude
    amplitude = x_array.max() - x_array.min()

    # Detect wave
    wave_detected = (
        num_zero_crossings >= min_zero_crossings and amplitude > amplitude_threshold
    )

    return wave_detected, velocity if wave_detected else None


def update_landmark_trail(
    current_trail: deque,
    mediapipe_results: Any,
    landmark_id: int,
    last_seen_time: Optional[float],
    current_time: float,
    timeout_duration: float = 2.0,
) -> Tuple[Optional[float], bool, Optional[float]]:
    """Update landmark trail with timeout management.

    Args:
        current_trail: Deque to store landmark coordinates
        mediapipe_results: MediaPipe results object
        landmark_id: Landmark index to track
        last_seen_time: Last time landmark was detected
        current_time: Current timestamp
        timeout_duration: Time after which to clear trail if no detection

    Returns:
        Tuple of (updated_last_seen_time, trail_was_cleared, extracted_coordinate)
    """
    trail_cleared = False
    extracted_coord = None

    if mediapipe_results and mediapipe_results.multi_hand_landmarks:
        hand_landmarks = mediapipe_results.multi_hand_landmarks[0]
        try:
            extracted_coord = hand_landmarks.landmark[landmark_id].x
            current_trail.append(extracted_coord)
            updated_last_seen_time = current_time
        except IndexError:
            logger.warning(f"Warning: Landmark index {landmark_id} is invalid.")
            updated_last_seen_time = last_seen_time
    else:
        updated_last_seen_time = last_seen_time
        if last_seen_time is not None and (
            current_time - last_seen_time > timeout_duration
        ):
            if current_trail:
                current_trail.clear()
                trail_cleared = True
            updated_last_seen_time = None

    return updated_last_seen_time, trail_cleared, extracted_coord


def close_mediapipe_hands():
    """Close all MediaPipe hands detector instances."""
    if hasattr(_thread_local, "hands_detector") and _thread_local.hands_detector:
        _thread_local.hands_detector.close()
        _thread_local.hands_detector = None


__all__ = [
    "get_mediapipe_hand_data",
    "draw_hand_landmarks",
    "is_wave_gesture",
    "update_landmark_trail",
    "get_hand_direction_and_flexion",
    "get_hand_pose_for_robot_control",
    "HandPoseTracker",
    "close_mediapipe_hands",
]


def get_hand_direction_and_flexion(
    hand_landmarks: Any,
    frame_shape: Tuple[int, ...],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Calculate hand direction and wrist flexion from landmarks.

    Uses wrist and finger MCP positions to determine:
    - Direction: 2D unit vector pointing from wrist toward fingers
    - Flexion: How much the hand is flexed (0 = flat, 1 = fully curled)

    Args:
        hand_landmarks: MediaPipe hand_landmarks object
        frame_shape: Shape of the input frame (h, w)

    Returns:
        Tuple of (direction_2d, flexion_ratio) or (None, None) if insufficient data
    """
    if not hand_landmarks or len(hand_landmarks.landmark) < 17:
        return None, None

    h, w = frame_shape[:2]

    wrist = hand_landmarks.landmark[WRIST]
    wrist_pos = np.array([wrist.x * w, wrist.y * h], dtype=np.float32)

    index_mcp = hand_landmarks.landmark[INDEX_FINGER_MCP]
    index_mcp_pos = np.array([index_mcp.x * w, index_mcp.y * h], dtype=np.float32)

    pinky_mcp = hand_landmarks.landmark[PINKY_MCP]
    pinky_mcp_pos = np.array([pinky_mcp.x * w, pinky_mcp.y * h], dtype=np.float32)

    palm_center = (index_mcp_pos + pinky_mcp_pos) / 2

    direction_2d = palm_center - wrist_pos
    direction_magnitude = np.linalg.norm(direction_2d)

    if direction_magnitude < 1e-6:
        return None, None

    direction_2d = direction_2d / direction_magnitude

    index_tip = hand_landmarks.landmark[INDEX_FINGER_TIP]
    index_tip_pos = np.array([index_tip.x * w, index_tip.y * h], dtype=np.float32)

    middle_tip = hand_landmarks.landmark[MIDDLE_FINGER_TIP]
    middle_tip_pos = np.array([middle_tip.x * w, middle_tip.y * h], dtype=np.float32)

    ring_tip = hand_landmarks.landmark[RING_FINGER_TIP]
    ring_tip_pos = np.array([ring_tip.x * w, ring_tip.y * h], dtype=np.float32)

    pinky_tip = hand_landmarks.landmark[PINKY_TIP]
    pinky_tip_pos = np.array([pinky_tip.x * w, pinky_tip.y * h], dtype=np.float32)

    tip_positions = np.array([index_tip_pos, middle_tip_pos, ring_tip_pos, pinky_tip_pos])
    mcp_positions = np.array([
        index_mcp_pos,
        np.array([hand_landmarks.landmark[MIDDLE_FINGER_MCP].x * w, hand_landmarks.landmark[MIDDLE_FINGER_MCP].y * h], dtype=np.float32),
        np.array([hand_landmarks.landmark[13].x * w, hand_landmarks.landmark[13].y * h], dtype=np.float32),  # Ring MCP = 13
        pinky_mcp_pos,
    ])

    curl_distances = np.linalg.norm(tip_positions - mcp_positions, axis=1)
    avg_curl = np.mean(curl_distances)

    max_curl = 80.0
    min_curl = 10.0
    flexion_ratio = np.clip((max_curl - avg_curl) / (max_curl - min_curl), 0.0, 1.0)

    return direction_2d, flexion_ratio


def get_hand_pose_for_robot_control(
    hand_landmarks: Any,
    frame_shape: Tuple[int, ...],
) -> Tuple[Optional[float], Optional[float]]:
    """Get hand pose for robot control: direction + wrist vertical position.

    - Hand direction: 2D angle from wrist toward fingers (controls robot rotation)
    - Wrist Y position: Vertical position of wrist (controls robot up/down)

    Args:
        hand_landmarks: MediaPipe hand_landmarks object
        frame_shape: Shape of the input frame (h, w)

    Returns:
        Tuple of (direction_angle_radians, wrist_y_normalized) or (None, None)
    """
    if not hand_landmarks or len(hand_landmarks.landmark) < 17:
        return None, None

    h, w = frame_shape[:2]

    wrist = hand_landmarks.landmark[WRIST]
    wrist_x = float(wrist.x * w)
    wrist_y = float(wrist.y * h)

    index_mcp = hand_landmarks.landmark[INDEX_FINGER_MCP]
    index_mcp_pos = np.array([index_mcp.x * w, index_mcp.y * h], dtype=np.float32)

    pinky_mcp = hand_landmarks.landmark[PINKY_MCP]
    pinky_mcp_pos = np.array([pinky_mcp.x * w, pinky_mcp.y * h], dtype=np.float32)

    palm_center = (index_mcp_pos + pinky_mcp_pos) / 2

    direction_2d = palm_center - np.array([wrist_x, wrist_y], dtype=np.float32)
    direction_magnitude = np.linalg.norm(direction_2d)

    if direction_magnitude < 1e-6:
        return None, None

    direction_angle = np.arctan2(direction_2d[1], direction_2d[0])

    wrist_y_normalized = 1.0 - (wrist_y / h)

    return direction_angle, wrist_y_normalized


class HandPoseTracker:
    """Tracks hand pose over time for robot control."""

    def __init__(self, smoothing_frames: int = 3):
        self.smoothing_frames = smoothing_frames
        self.angle_history: Deque[float] = deque(maxlen=smoothing_frames)
        self.wrist_y_history: Deque[float] = deque(maxlen=smoothing_frames)
        self.initial_angle: Optional[float] = None
        self.initial_wrist_y: Optional[float] = None
        self.is_calibrated = False

    def update(
        self,
        direction_angle: Optional[float],
        wrist_y: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Update tracker with new pose data.

        Returns:
            Tuple of (smoothed_angle, smoothed_wrist_y, angle_delta, wrist_y_delta)
            - smoothed_angle: Averaged direction angle
            - smoothed_wrist_y: Averaged wrist Y position
            - angle_delta: Change in angle since calibration (for robot rotation)
            - wrist_y_delta: Change in wrist Y since calibration (for robot up/down)
        """
        if direction_angle is not None:
            self.angle_history.append(direction_angle)

        if wrist_y is not None:
            if not self.is_calibrated:
                self.initial_angle = direction_angle
                self.initial_wrist_y = wrist_y
                self.is_calibrated = True
            self.wrist_y_history.append(wrist_y)

        if len(self.angle_history) == 0 or len(self.wrist_y_history) == 0:
            return None, None, None, None

        smoothed_angle = np.mean(list(self.angle_history))
        smoothed_wrist_y = np.mean(list(self.wrist_y_history))

        angle_delta = 0.0
        wrist_y_delta = 0.0
        if self.initial_angle is not None and self.initial_wrist_y is not None:
            angle_delta = smoothed_angle - self.initial_angle
            wrist_y_delta = smoothed_wrist_y - self.initial_wrist_y

        return smoothed_angle, smoothed_wrist_y, angle_delta, wrist_y_delta

    def reset_calibration(self):
        """Reset calibration to capture new baseline."""
        self.is_calibrated = False
        self.initial_angle = None
        self.initial_wrist_y = None
        self.angle_history.clear()
        self.wrist_y_history.clear()


def extract_hand_features(hand_landmarks: Any, frame_shape: Tuple[int, ...]) -> Optional[dict]:
    """Extract all hand features based on the custom logic mapper diagrams."""
    if not hand_landmarks or len(hand_landmarks.landmark) < 21:
        return None

    lm = hand_landmarks.landmark
    h, w = frame_shape[:2]

    # 1. Wrist Y position -> Bend front/back
    wrist = lm[0]
    wrist_y = wrist.y

    # 2. Palm normal X -> Bend left/right
    # using index_mcp(5), pinky_mcp(17), wrist(0)
    # Vectors strictly from wrist to MCPs
    index_mcp = lm[5]
    pinky_mcp = lm[17]
    u = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    normal = np.cross(u, v)
    normal_mag = np.linalg.norm(normal)
    palm_normal_x = normal[0] / normal_mag if normal_mag > 1e-6 else 0.0

    # 3. Hand bounding box -> Robot extension
    xs = [pt.x for pt in lm]
    ys = [pt.y for pt in lm]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

    # 4. In-plane roll angle -> Robot base yaw
    dx = index_mcp.x - pinky_mcp.x
    dy = index_mcp.y - pinky_mcp.y
    roll_angle = np.arctan2(dy, dx)

    # 5. Normalised ratio -> Gripper latch
    # user logic: gap = dist(thumb_tip, index_tip), palm_diag = dist(wrist, pinky_mcp)
    thumb_tip = lm[4]
    index_tip = lm[8]
    def dist(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    gap = dist(thumb_tip, index_tip)
    palm_diag = dist(wrist, pinky_mcp)
    pinch_ratio = gap / (palm_diag + 1e-6)

    # 6. Fingertip Z variance -> Override / Neutral
    z_tips = [lm[4].z, lm[8].z, lm[12].z, lm[16].z, lm[20].z]
    z_variance = float(np.var(z_tips))

    # 7. Wrist X position (stable horizontal anchor for L/R robot control)
    wrist_x = wrist.x  # normalised [0,1]; left=0, right=1 in mirrored frame

    # 8. Thumb-to-middle-finger distance (unlock gesture)
    middle_tip = lm[12]
    thumb_middle_gap = dist(thumb_tip, middle_tip)
    thumb_middle_ratio = thumb_middle_gap / (palm_diag + 1e-6)

    return {
        "wrist_y": wrist_y,
        "wrist_x": wrist_x,
        "palm_normal_x": palm_normal_x,
        "bbox_area": bbox_area,
        "roll_angle": roll_angle,
        "pinch_ratio": pinch_ratio,
        "thumb_middle_ratio": thumb_middle_ratio,
        "z_variance": z_variance
    }


# Rex Tendon Hand Mapping Redesign

This plan covers the rewrite of the `hand_tracking.py` and `hand_sim_controller.py` modules to strictly follow the logic diagrams provided for controlling the robot using MediaPipe landmarks.

## Implementation Details

### 1. Feature Extraction Upgrade (`hand_tracking.py`)
We will replace the existing simplistic `get_hand_pose_for_robot_control` and `get_hand_direction_and_flexion` functions with a new comprehensive parsing function: `extract_hand_features(hand_landmarks, frame_shape)`.

It will calculate:
*   **Wrist Y position**: Direct normalization of `landmark[0].y`.
*   **Palm Normal X**: Compute relative vectors `u = index_mcp - wrist` and `v = pinky_mcp - wrist`. Normal vector = `cross(u, v)`. Return `normal.x`.
*   **Bounding Box Area**: Compute max - min along X and Y for all landmarks, multiply to get the bounding box area in normalized image space.
*   **In-plane Roll (Yaw)**: `atan2(index_mcp.y - pinky_mcp.y, index_mcp.x - pinky_mcp.x)`.
*   **Pinch Ratio**: `dist(thumb_tip, index_tip) / dist(wrist, index_mcp)`.
*   **Fingertip Z Variance**: `variance([thumb.z, index.z, middle.z, ring.z, pinky.z])`.

### 2. Controller Mapping Logic (`hand_sim_controller.py`)
We will rewrite the main loop taking the variables above and passing them through the state machine from the diagrams.

#### Cursor Mapping (Bend Front/Back/Left/Right + Yaw)
*   `cursor_y` (Front/Back) = Mapped from the vertical wrist Y.
*   `cursor_x` (Left/Right) = Mapped from Palm Normal X.
*   The raw `[cursor_x, cursor_y]` vector will be rotated by the extracted `roll_angle` minus an initial calibrated `base_roll_angle`. This implements the **"Robot base yaw decoupled from tilt"** mechanism using the existing 3 tendon limits.

#### Depth/Extension (Bounding Box)
*   Map the Bounding Box Area inversely to the target tendon baseline: `small = far = extend (low baseline)`, `large = close = retract (high baseline)`.

#### Gripper State Machine
Implement the exact thresholds shown in the second flowchart:
*   `ratio < 0.25`: TRIGGER GRAB LOCK.
*   `ratio > 0.60`: TRIGGER GRAB RELEASE.
*   Maintain last state for anything in between (hysteresis).
*   Lose tracking: Hold last state (no dropout releases).

#### Override State
*   If `fingertip_z_variance < OVERRIDE_THRESHOLD` (meaning a flat, coplanar hand), zero out the cursor and reset the baseline to neutral (0.23).

### 3. Calibration
We will keep the `c` key to calibrate the "neutral" zero-points of the mapping. Pressing `c` will record the current wrist Y, palm normal X, and roll angle to center future movements.

## User Review Required
> [!IMPORTANT]
> The robot does not have a physical "Base Yaw" twisting actuator in the simulation, it only has 3 linear tendons that bend it.
> As designed in the plan above, "Yawing" the wrist will digitally rotate the target frame of the `convert_2d_cursor_to_target_lengths` matrix (effectively causing the robot's bending to arc/yaw around its vertical axis). This is the standard way to achieve base yaw on a tendon column. Please confirm if this matches your intention!

## Verification Plan
1. Start `run_hand_control.py` and raise arm in neutral pose. Press `c` to calibrate.
2. **Move up/down**: Observe purely vertical bending.
3. **Tilt palm left/right (supination/pronation)**: Observe purely lateral bending.
4. **Push/pull hand to camera**: Observe Z-axis extension/retraction.
5. **Twist wrist CW/CCW (yaw)**: Observe the bending vector rotating.
6. **Pinch thumb/index**: Observe the GRASP LOCK HUD status and logic.
7. **Hold flat palm**: Observe robot snapping smoothly back to neutral.

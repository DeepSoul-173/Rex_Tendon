"""Geometry utilities for cursor to motor position conversion."""

import numpy as np
from typing import Dict, Tuple
from ..common.constants import (
    MOTOR_NORMALIZED_POSITIONS,
    MOTOR_ONE_FULL_TURN_TICKS,
    MOTOR_POSITION_MIN,
)


def cursor_to_motor_positions(
    cursor_pos: np.ndarray,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
    cursor_deadzone: float = 0.01,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Convert 2D cursor position to motor tick positions.

    This is the canonical function for converting normalized 2D coordinates
    to motor commands. It's used by all higher-level components including
    trackpad control, primitives, and RL agents.

    Args:
        cursor_pos: Normalized 2D cursor position as np.array([x, y])
                   where x, y are in range [-1, 1]
        calibrated_ticks_map: Dictionary mapping motor names to calibrated positions
        noise_scale: Optional noise to add to cursor position for randomization
        cursor_deadzone: Minimum cursor movement to register (default: 0.01)

    Returns:
        Tuple of (target_positions, position_offsets) where:
        - target_positions: Dict mapping motor names to absolute tick positions
        - position_offsets: Dict mapping motor names to offset from calibrated position
    """
    target_positions = {}
    calculated_offsets = {}

    # Add noise if specified (useful for training data augmentation)
    if noise_scale > 0.0:
        noise = np.random.normal(0, noise_scale, cursor_pos.shape)
        cursor_pos = cursor_pos + noise

    # Apply deadzone to avoid jitter at center
    cursor_magnitude = np.linalg.norm(cursor_pos)
    if cursor_magnitude < cursor_deadzone:
        # Return calibrated positions for very small movements
        return calibrated_ticks_map.copy(), {name: 0 for name in calibrated_ticks_map}

    # Normalize cursor direction
    cursor_dir = cursor_pos / cursor_magnitude

    # Calculate effect for each motor based on alignment with cursor direction
    for motor_name, motor_pos_normalized in MOTOR_NORMALIZED_POSITIONS.items():
        # Get calibrated start position for this motor
        calibrated_start_pos = calibrated_ticks_map.get(motor_name, 0)

        # Calculate motor direction vector
        motor_magnitude = np.linalg.norm(motor_pos_normalized)
        if motor_magnitude > 0.01:  # Avoid division by zero
            motor_dir = motor_pos_normalized / motor_magnitude
        else:
            motor_dir = np.array([0.0, 0.0])

        # Calculate alignment between cursor direction and motor direction
        # This determines how much this motor should move
        alignment = np.dot(cursor_dir, motor_dir)

        # Scale alignment by cursor magnitude to get motor effect
        # Motors aligned with cursor direction pull more, opposing motors pull less
        effect = alignment * cursor_magnitude

        effect = max(-1.0, min(1.0, effect))

        # Convert effect to tick offset
        position_offset = int(effect * MOTOR_ONE_FULL_TURN_TICKS)

        # Calculate absolute target position
        absolute_position = calibrated_start_pos + position_offset

        # Handle wrap-around for positions outside valid range
        # This allows for continuous rotation
        if absolute_position >= 0:
            target_position = absolute_position
        else:
            # Wrap negative positions using two's complement representation
            target_position = MOTOR_POSITION_MIN - absolute_position

        target_positions[motor_name] = target_position
        calculated_offsets[motor_name] = position_offset

    return target_positions, calculated_offsets


def convert_2d_cursor_to_target_lengths(
    cursor_pos_2d: np.ndarray,
    baseline_lengths_m: np.ndarray,
    actuator_low_lengths_m: np.ndarray,
    actuator_high_lengths_m: np.ndarray,
    max_2d_action_magnitude: float = 1.0,
) -> np.ndarray:
    """Convert a 2-D cursor action into three tendon lengths.

    This logic was previously duplicated across closed_loop.py and the RL
    training environment.  It now lives in geometry.py so it can be shared.

    Args:
        cursor_pos_2d: Normalised 2-D cursor coordinates ``[x, y]`` in range
            ``[-1, 1]``.
        baseline_lengths_m: Baseline (relaxed) tendon lengths.
        actuator_low_lengths_m: Minimum allowed tendon lengths.
        actuator_high_lengths_m: Maximum allowed tendon lengths.
        max_2d_action_magnitude: Maximum magnitude expected for *cursor_pos_2d*.

    Returns:
        (3,) numpy array of target lengths in metres for the three simulated
        tendons.
    """
    target_lengths_m = np.zeros(3, dtype=np.float32)

    cursor_pos_2d = np.clip(
        cursor_pos_2d, -max_2d_action_magnitude, max_2d_action_magnitude
    )

    for i in range(3):
        sim_tendon_dir_normalized = MOTOR_NORMALIZED_POSITIONS[str(i + 1)]
        baseline_len = baseline_lengths_m[i]
        act_low_len = actuator_low_lengths_m[i]
        act_high_len = actuator_high_lengths_m[i]

        if np.linalg.norm(cursor_pos_2d) > 0.01:
            cursor_dir = cursor_pos_2d / np.linalg.norm(cursor_pos_2d)
            norm_tendon_dir = np.linalg.norm(sim_tendon_dir_normalized)
            if norm_tendon_dir > 0.01:
                tendon_dir = sim_tendon_dir_normalized / norm_tendon_dir
            else:
                tendon_dir = np.array([0.0, 0.0])
            alignment = float(np.dot(cursor_dir, tendon_dir))
        else:
            alignment = 0.0

        effect = alignment * float(np.linalg.norm(cursor_pos_2d))
        if max_2d_action_magnitude > 1e-6:
            effect = effect / max_2d_action_magnitude
        else:
            effect = 0.0
        effect = np.clip(effect, -1.0, 1.0)

        if effect > 0:
            length_change = effect * (baseline_len - act_low_len)
        else:
            length_change = effect * (act_high_len - baseline_len)

        target_length = baseline_len - length_change
        target_lengths_m[i] = np.clip(target_length, act_low_len, act_high_len)

    return target_lengths_m


def convert_4d_cursor_to_target_lengths(
    cursor_4d: np.ndarray,
    baseline_lengths_m: np.ndarray,
    actuator_low_lengths_m: np.ndarray,
    actuator_high_lengths_m: np.ndarray,
    max_2d_action_magnitude: float = 1.0,
) -> np.ndarray:
    """Convert a 4-D action into six tendon lengths for a 2-section arm.

    The action is two stacked 2-D cursors, ``[s1_x, s1_y, s2_x, s2_y]``: the first
    drives the lower section (tendons 1-3), the second the upper section (tendons
    4-6).  Each section reuses the single-section 2-D cursor -> 3-tendon map, so
    the two halves bend independently (4 task-DOF, enabling S-curves).

    Args:
        cursor_4d: ``[s1_x, s1_y, s2_x, s2_y]`` in ``[-max, max]``.
        baseline_lengths_m: (6,) baseline tendon lengths (3 per section).
        actuator_low_lengths_m / actuator_high_lengths_m: (6,) per-tendon bounds.
        max_2d_action_magnitude: max magnitude of each 2-D cursor.

    Returns:
        (6,) numpy array of target tendon lengths in metres.
    """
    cursor_4d = np.asarray(cursor_4d, dtype=np.float32).reshape(-1)
    baseline_lengths_m = np.asarray(baseline_lengths_m, dtype=np.float32)
    actuator_low_lengths_m = np.asarray(actuator_low_lengths_m, dtype=np.float32)
    actuator_high_lengths_m = np.asarray(actuator_high_lengths_m, dtype=np.float32)

    section_lengths = []
    for s in range(2):
        cursor = cursor_4d[2 * s : 2 * s + 2]
        lo, hi = 3 * s, 3 * s + 3
        section_lengths.append(
            convert_2d_cursor_to_target_lengths(
                cursor,
                baseline_lengths_m[lo:hi],
                actuator_low_lengths_m[lo:hi],
                actuator_high_lengths_m[lo:hi],
                max_2d_action_magnitude,
            )
        )
    return np.concatenate(section_lengths).astype(np.float32)


"""Control configuration for closed-loop control workflows."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import Field
from .base import BaseConfig


class ControlConfig(BaseConfig):
    """Configuration for closed-loop control workflows."""

    # RL Control Safety Parameters
    safety_offset_min: int = Field(
        default=-4000, description="RL control: Min tick offset from calibrated zero"
    )
    safety_offset_max: int = Field(
        default=4000,
        description="RL control: Max tick offset from calibrated zero (pull direction)",
    )

    # RL Configuration
    model_path: str = Field(default="", description="Path to trained RL model")
    num_frames: int = Field(default=4, description="Number of observation frames")
    deterministic: bool = Field(
        default=True, description="Use deterministic policy actions"
    )
    actuator_low: float = Field(
        default=0.12, description="Minimum actuator length in metres"
    )
    actuator_high: float = Field(
        default=0.34, description="Maximum actuator length in metres"
    )
    max_2d_action_magnitude: float = Field(
        default=1.0, description="Maximum magnitude for 2D cursor actions"
    )
    include_actuator_lengths_in_obs: bool = Field(
        default=True, description="Whether to include actuator lengths in observation"
    )
    clip_observations: bool = Field(
        default=True, description="Master switch for observation clipping"
    )

    obs_clip_bounds: Dict[str, Any] = Field(
        default={
            "tip_min": [-0.5, -0.3, 0.00],
            "tip_max": [-0.0, 0.3, 0.3],
            "target_min": [-0.2, -0.06, 0.12],
            "target_max": [-0.1, 0.08, 0.22],
            "actuator_length_min": 0.12,
            "actuator_length_max": 0.28,
        },
        description="Observation clipping bounds",
    )

    # Calibration Configuration - Direct 2D Position
    calibration_2d_position: Tuple[float, float] = Field(
        default=(0.0, -0.3),
        description="2D position for calibration in control plane [x, y] in range [-1, 1]",
    )

    # Orchestrator Configuration
    lost_frames_threshold: int = Field(
        default=15, description="Consecutive missing frames before recovery"
    )
    stop_on_prolonged_loss: bool = Field(
        default=False,
        description="Stop controller entirely instead of returning to home when finger lost for too long",
    )

    # Action Smoothing Configuration
    action_smoothing_alpha: float = Field(
        default=0.3,
        description="Action smoothing factor (0.0=max smoothing, 1.0=no smoothing)",
    )

    # Action Calibration Configuration
    action_2d_offset: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Constant 2D offset [x, y] added to RL action output for calibration",
    )

    # Idle Motion Configuration
    idle_motion_pattern_config: Dict[str, Any] = Field(
        default={
            "base_period": 1.0,
            "base_period_range": [1.0, 5.0],
            "amplitude": 0.05,
            "amplitude_range": [0.05, 0.3],
            "direction_jitter_interval": 5.0,
            "direction_jitter_interval_range": [5.0, 20.0],
            "direction_change_duration": 10.0,
            "origin_avoidance_radius": 0.07,
            "frequency_transition_duration": 10.0,
            "jitter_frequency_transition_duration": 10,
            "jitter": 0.0,
        },
        description="Configuration parameters for idle motion breathing pattern",
    )
    idle_motion_max_motor_step_per_loop: int = Field(
        default=30,
        description="Maximum motor step size per loop iteration for idle motion",
    )


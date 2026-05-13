"""Hardware configuration for motors and devices."""

from pathlib import Path
from typing import Dict, List
from pydantic import Field
from .base import BaseConfig


class HardwareConfig(BaseConfig):
    """Configuration for hardware components."""

    # Motor configuration
    motor_config: Dict[str, List] = Field(
        default={
            "1": [1, "sts3215"],  # [id, model]
            "2": [2, "sts3215"],
            "3": [3, "sts3215"],
        },
        description="Motor configuration mapping motor names to [id, model]",
    )
    baudrate: int = Field(default=1000000, description="Serial baudrate")

    # Serial port configuration
    port: str = Field(default="", description="Serial port for motor communication")

    # Calibration
    calibration_file: Path = Field(
        default=Path("rex_assets/rex_hardware/calibration/tentacle_calibration.json"),
        description="Path to motor calibration file",
    )

    # Motor physics parameters
    ticks_per_rotation: int = Field(
        default=4096, description="Motor ticks per full rotation"
    )
    length_per_rotation: float = Field(
        default=0.11, description="Metres per full spool rotation"
    )
    baseline_length: float = Field(
        default=0.23, description="Relaxed cable length in metres"
    )
    tick_sign: int = Field(
        default=1,
        description="+1: increasing ticks lengthen cable, -1: increasing ticks shorten",
    )

    # Control parameters (moved from constants)
    motor_settle_time: float = Field(
        default=0.5, description="Time to wait for motors to reach position (seconds)"
    )
    position_tolerance: int = Field(
        default=100, description="Acceptable position error in ticks"
    )

    # Component timing parameters
    control_loop_hz: float = Field(
        default=50.0, description="Main motor control loop frequency (Hz)"
    )
    trackpad_control_hz: float = Field(
        default=50.0, description="Trackpad control update rate (Hz)"
    )
    idle_motion_hz: float = Field(
        default=50.0, description="Idle motion pattern update rate (Hz)"
    )


"""Motion primitives and behaviors (ported from legacy action_normalized.py)."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import numpy as np
import typer
from rich.console import Console

from .geometry import cursor_to_motor_positions
from ..hardware.motors import MotorController
from ..common.constants import MOTOR_NORMALIZED_POSITIONS
from ..configs import get_hardware_config

console = Console()
app = typer.Typer(help="Motion primitive utilities")
logger = logging.getLogger(__name__)


class MotionBehavior(Enum):
    """Enumeration of available motion behaviors."""

    YES = "<yes>"
    NO = "<no>"
    SHAKE = "<shake>"
    CIRCLE = "<circle>"
    GRAB = "<grab_object>"
    RELEASE = "<release_object>"
    HIGH_FIVE = "<high_five>"

    @classmethod
    def from_action_string(cls, action_string: str) -> Optional["MotionBehavior"]:
        """Get behavior from action string.

        Args:
            action_string: The action string (e.g., "<yes>", "<grab_object>")

        Returns:
            MotionBehavior enum value if found, None otherwise
        """
        try:
            return cls(action_string)
        except ValueError:
            return None


@dataclass
class YesMotionConfig:
    """Configuration for yes/nodding motion - handcrafted for natural movement."""

    down_position: np.ndarray = field(default_factory=lambda: np.array([0.12, -0.08]))
    center_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.13


@dataclass
class NoMotionConfig:
    """Configuration for no/head-shake motion - handcrafted for natural movement."""

    left_position: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.0]))
    down_position: np.ndarray = field(default_factory=lambda: np.array([0.12, -0.08]))
    right_position: np.ndarray = field(default_factory=lambda: np.array([-0.0, -0.15]))
    initial_delay: float = 0.05
    hold_duration: float = 0.13


@dataclass
class ShakeMotionConfig:
    """Configuration for shake motion - handcrafted for natural movement."""

    left_position: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.07]))
    right_position: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.07]))
    hold_duration: float = 0.17


@dataclass
class CircleMotionConfig:
    """Configuration for circular motion - handcrafted for smooth movement."""

    radius: float = 0.07
    points_per_circle: int = 20
    time_per_point: float = 0.009


@dataclass
class GrabMotionConfig:
    """Configuration for grab/release motions - handcrafted positions."""

    grab_cursor_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.061, -0.035])
    )
    hold_duration: float = 0.3


@dataclass
class ReleaseMotionConfig:
    """Configuration for release motion - handcrafted for natural movement."""

    neutral_cursor_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.3


@dataclass
class HighFiveMotionConfig:
    """Configuration for high five motion - handcrafted for natural movement."""

    high_five_position: np.ndarray = field(
        default_factory=lambda: np.array([0.10392, -0.06])
    )
    center_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.06


YES_CONFIG = YesMotionConfig()
NO_CONFIG = NoMotionConfig()
SHAKE_CONFIG = ShakeMotionConfig()
CIRCLE_CONFIG = CircleMotionConfig()
GRAB_CONFIG = GrabMotionConfig()
RELEASE_CONFIG = ReleaseMotionConfig()
HIGH_FIVE_CONFIG = HighFiveMotionConfig()


def perform_yes_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform yes/nodding motion."""
    for _ in range(4):
        target_positions_down, _ = cursor_to_motor_positions(
            cursor_pos=YES_CONFIG.down_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_down)
        time.sleep(YES_CONFIG.hold_duration)

        target_positions_centre, _ = cursor_to_motor_positions(
            cursor_pos=YES_CONFIG.center_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_centre)
        time.sleep(YES_CONFIG.hold_duration)


def perform_no_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform no/head-shake motion."""
    target_positions_down, _ = cursor_to_motor_positions(
        cursor_pos=NO_CONFIG.down_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_down)
    time.sleep(NO_CONFIG.initial_delay)

    for _ in range(4):
        target_positions_left, _ = cursor_to_motor_positions(
            cursor_pos=NO_CONFIG.left_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_left)
        time.sleep(NO_CONFIG.hold_duration)

        target_positions_right, _ = cursor_to_motor_positions(
            cursor_pos=NO_CONFIG.right_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_right)
        time.sleep(NO_CONFIG.hold_duration)


def perform_shake_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform shake motion."""
    for _ in range(4):
        target_positions_left, _ = cursor_to_motor_positions(
            cursor_pos=SHAKE_CONFIG.left_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_left)
        time.sleep(SHAKE_CONFIG.hold_duration)

        target_positions_right, _ = cursor_to_motor_positions(
            cursor_pos=SHAKE_CONFIG.right_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_right)
        time.sleep(SHAKE_CONFIG.hold_duration)


def perform_circle_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform circular motion in XY plane."""
    for _ in range(4):
        for i in range(CIRCLE_CONFIG.points_per_circle):
            angle = (i / CIRCLE_CONFIG.points_per_circle) * 2 * np.pi
            cursor_pos = np.array(
                [
                    CIRCLE_CONFIG.radius * np.cos(angle),
                    CIRCLE_CONFIG.radius * np.sin(angle),
                ]
            )

            target_positions, _ = cursor_to_motor_positions(
                cursor_pos=cursor_pos,
                calibrated_ticks_map=calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            motor_controller.set_positions(target_positions)
            time.sleep(CIRCLE_CONFIG.time_per_point)


def perform_grab_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Move tentacle to predefined grabbing position."""
    logger.info("Moving to GRAB position: %s", GRAB_CONFIG.grab_cursor_pos)
    target_positions_grab, _ = cursor_to_motor_positions(
        cursor_pos=GRAB_CONFIG.grab_cursor_pos,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_grab)
    time.sleep(GRAB_CONFIG.hold_duration)


def perform_release_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Move tentacle to neutral position, releasing grab."""
    logger.info("Moving to NEUTRAL position: %s", RELEASE_CONFIG.neutral_cursor_pos)
    target_positions_neutral, _ = cursor_to_motor_positions(
        cursor_pos=RELEASE_CONFIG.neutral_cursor_pos,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_neutral)
    time.sleep(RELEASE_CONFIG.hold_duration)


def perform_high_five_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform high five motion."""
    # Move to high five position
    target_positions_high_five, _ = cursor_to_motor_positions(
        cursor_pos=HIGH_FIVE_CONFIG.high_five_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_high_five)
    time.sleep(HIGH_FIVE_CONFIG.hold_duration)

    # Return to center position
    target_positions_centre, _ = cursor_to_motor_positions(
        cursor_pos=HIGH_FIVE_CONFIG.center_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_centre)
    time.sleep(HIGH_FIVE_CONFIG.hold_duration)


def execute_behavior(
    motor_controller: MotorController,
    behavior: MotionBehavior,
    *,
    noise_scale: float = 0.010,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a motion behavior primitive.

    Args:
        motor_controller: Connected motor controller
        behavior: The motion behavior to execute
        noise_scale: Scale of random noise to apply
        **kwargs: Additional behavior-specific parameters

    Returns:
        Dictionary with execution result information
    """
    if not motor_controller.is_connected:
        return {
            "behavior": behavior.value,
            "status": "error",
            "message": "Motor controller not connected",
        }

    # Get calibration data
    calibrated_ticks_map = motor_controller.get_calibration_data()

    behaviors_performed = False
    reset_after_sequence = False

    try:
        if behavior == MotionBehavior.YES:
            perform_yes_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.NO:
            perform_no_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SHAKE:
            perform_shake_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.CIRCLE:
            perform_circle_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.GRAB:
            perform_grab_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = False

        elif behavior == MotionBehavior.RELEASE:
            perform_release_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.HIGH_FIVE:
            perform_high_five_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        # Handle reset logic
        if behaviors_performed and reset_after_sequence:
            logger.info("Behaviors complete. Resetting motors to zero")
            motor_controller.reset_to_calibrated_positions()
            time.sleep(0.2)
        elif behaviors_performed and not reset_after_sequence:
            logger.info("Grab behavior complete. Motors will remain in grab pose")

        return {
            "behavior": behavior.value,
            "status": "success",
            "message": f"Successfully executed {behavior.value} behavior",
            "reset_performed": reset_after_sequence,
        }

    except Exception as e:
        logger.error("Error executing behavior %s: %s", behavior.value, e)
        return {
            "behavior": behavior.value,
            "status": "error",
            "message": f"Error executing behavior: {e}",
        }


@app.command()
def run(
    behavior: str = typer.Argument(
        help="Motion behavior to test: yes, no, shake, circle, grab, release, high_five"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    noise_scale: float = typer.Option(
        0.010, "--noise", "-n", help="Noise scale for motion randomization"
    ),
) -> None:
    """Test a motion primitive behavior."""

    console.print(f"[bold blue]Testing Motion Primitive: {behavior}[/bold blue]")

    try:
        # Validate behavior
        try:
            motion_behavior = MotionBehavior(behavior)
        except ValueError:
            console.print(f"[red]Error: Unknown behavior '{behavior}'[/red]")
            console.print("[yellow]Available behaviors:[/yellow]")
            for b in MotionBehavior:
                console.print(f"  • {b.value}")
            raise typer.Exit(1)

        # Create config from file
        hardware_config = get_hardware_config(config)

        console.print(
            f"Connecting to motors on port: [cyan]{hardware_config.port}[/cyan]"
        )

        # Connect to motors
        with console.status("[bold green]Connecting..."):
            motor_controller = MotorController(hardware_config)
            motor_controller.connect()

        console.print("[green]✓[/green] Connected to motors")

        console.print(f"[bold yellow]Executing {behavior} behavior...[/bold yellow]")

        result = execute_behavior(
            motor_controller=motor_controller,
            behavior=motion_behavior,
            noise_scale=noise_scale,
        )

        if result["status"] == "success":
            console.print(f"[green]✓[/green] {result['message']}")
            if result.get("reset_performed"):
                console.print("[dim]Motors reset to calibrated positions[/dim]")
        else:
            console.print(f"[red]Error: {result['message']}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


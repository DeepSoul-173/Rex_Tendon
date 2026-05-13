"""Trackpad control interface for tentacle motor control."""

import time
from typing import Optional
import numpy as np
from pynput import mouse
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import typer
import logging

from ..hardware.motors import MotorController
from ..configs.loaders import get_hardware_config
from ..control.geometry import cursor_to_motor_positions

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Trackpad control utility")


class TrackpadController:
    """Controller for trackpad/mouse input with calibration support."""

    def __init__(self):
        """Initialize trackpad controller."""
        self.x: float = 0.0
        self.y: float = 0.0
        self.running: bool = True
        self.listener = None

        # Calibration bounds
        self.min_x: float = float("inf")
        self.max_x: float = float("-inf")
        self.min_y: float = float("inf")
        self.max_y: float = float("-inf")

        # Calibrated center
        self.center_x: Optional[float] = None
        self.center_y: Optional[float] = None

    def on_move(self, x: float, y: float) -> None:
        """Handle cursor movement event.

        Args:
            x, y: Cursor coordinates
        """
        self.x = x
        self.y = y

        # Update calibration bounds
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def start(self) -> None:
        """Start listening to trackpad events."""
        try:
            self.listener = mouse.Listener(on_move=self.on_move)
            self.listener.start()
        except ImportError:
            raise ImportError(
                "pynput library required for trackpad control. "
                "Install with: pip install pynput"
            )

    def stop(self) -> None:
        """Stop listening to trackpad events."""
        if self.listener:
            self.listener.stop()
            self.listener = None

    def calibrate(self, duration: float = 2.0) -> None:
        """Calibrate trackpad bounds by having user move cursor around.

        Args:
            duration: Calibration duration in seconds
        """
        logger.info("Calibrating trackpad for %.1f seconds", duration)
        console.print("Please move your cursor to all FOUR edges of the trackpad")

        start_time = time.time()
        with Live(
            "[yellow]Calibrating...[/yellow]", console=console, refresh_per_second=10
        ) as live:
            while time.time() - start_time < duration:
                remaining = duration - (time.time() - start_time)
                live.update(
                    f"[yellow]Calibrating... {remaining:.1f}s remaining[/yellow]"
                )
                time.sleep(0.1)

        # Calculate center
        self.center_x = (self.min_x + self.max_x) / 2
        self.center_y = (self.min_y + self.max_y) / 2

        logger.info("Calibration complete!")
        logger.info(
            "X range: %.0f to %.0f, center: %.0f", self.min_x, self.max_x, self.center_x
        )
        logger.info(
            "Y range: %.0f to %.0f, center: %.0f", self.min_y, self.max_y, self.center_y
        )

    def get_normalized_position(
        self, screen_width: float, screen_height: float
    ) -> np.ndarray:
        """Get normalized cursor position in [-1, 1] range.

        Args:
            screen_width: Screen width for fallback calibration
            screen_height: Screen height for fallback calibration

        Returns:
            Normalized position as np.array([norm_x, norm_y]) in range [-1, 1]
        """
        # Use calibrated center if available, otherwise use screen center
        if self.center_x is None:
            self.center_x = screen_width / 2
            self.center_y = screen_height / 2

        # Calculate range from center to edges
        x_range = max(
            (
                self.max_x - self.center_x
                if self.max_x != float("-inf")
                else screen_width / 2
            ),
            (
                self.center_x - self.min_x
                if self.min_x != float("inf")
                else screen_width / 2
            ),
        )
        y_range = max(
            (
                self.max_y - self.center_y
                if self.max_y != float("-inf")
                else screen_height / 2
            ),
            (
                self.center_y - self.min_y
                if self.min_y != float("inf")
                else screen_height / 2
            ),
        )

        # Ensure minimum range to avoid division by zero
        if x_range < 50:
            x_range = screen_width / 2
        if y_range < 50:
            y_range = screen_height / 2

        # Normalize to [-1, 1] range
        norm_x = (self.x - self.center_x) / x_range
        norm_y = (self.y - self.center_y) / y_range

        # Clamp to valid range
        norm_x = max(-1.0, min(1.0, norm_x))
        norm_y = max(-1.0, min(1.0, norm_y))

        return np.array([norm_x, norm_y])

    def reset_calibration(self) -> None:
        """Reset calibration bounds."""
        self.min_x = float("inf")
        self.max_x = float("-inf")
        self.min_y = float("inf")
        self.max_y = float("-inf")
        self.center_x = None
        self.center_y = None


def control_with_trackpad(config: Optional[str] = None) -> None:
    """Control tentacle motors using trackpad input.

    Args:
        config: Optional path to configuration file
    """
    console.print("[bold blue]Trackpad Control Tool[/bold blue]")
    console.print("Use your trackpad to control the tentacle motors.")
    console.print("Press [red]Ctrl+C[/red] to exit.\n")

    # Create hardware config from file
    hardware_config = get_hardware_config(config)

    try:
        # Connect to motors
        with console.status("[bold green]Connecting to motors..."):
            motor_controller = MotorController(hardware_config)
            motor_controller.connect()

        console.print("[green]✓[/green] Connected to motors successfully")

        # Initialize trackpad controller
        trackpad = TrackpadController()
        trackpad.start()

        # Calibrate trackpad
        console.print("\n[yellow]Calibrating trackpad...[/yellow]")
        trackpad.calibrate(duration=10.0)

        # Get calibration data
        calibration_data = motor_controller.get_calibration_data()

        console.print("\n[green]Starting trackpad control...[/green]")
        console.print("Move your cursor around the trackpad to control the tentacle")

        # Main control loop with live display
        with Live(generate_status_display(), refresh_per_second=10) as live:
            try:
                while True:
                    # Get normalized trackpad position directly (no double normalization!)
                    cursor_pos = trackpad.get_normalized_position(1000, 1000)

                    # Convert to motor positions
                    target_positions, offsets = cursor_to_motor_positions(
                        cursor_pos,
                        calibration_data,
                        0.0,  # Use config for noise_scale if needed
                    )

                    # Send commands to motors
                    motor_controller.set_positions(target_positions)

                    # Update display
                    live.update(
                        generate_status_display(
                            cursor_pos=cursor_pos,
                            target_positions=target_positions,
                            offsets=offsets,
                            trackpad_pos=cursor_pos,  # Same as cursor_pos now
                        )
                    )

                    # Use configurable update rate from hardware config
                    sleep_time = 1.0 / hardware_config.trackpad_control_hz
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping trackpad control...[/yellow]")

        # Reset to calibrated positions
        console.print("[yellow]Resetting motors to calibrated positions...[/yellow]")
        motor_controller.reset_to_calibrated_positions()

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        if "trackpad" in locals():
            trackpad.stop()


def generate_status_display(
    cursor_pos=None, target_positions=None, offsets=None, trackpad_pos=None
) -> Panel:
    """Generate live status display."""
    table = Table(title="Trackpad Control Status")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    if trackpad_pos is not None:
        table.add_row("Trackpad X", f"{trackpad_pos[0]:+.3f}")
        table.add_row("Trackpad Y", f"{trackpad_pos[1]:+.3f}")

    if cursor_pos is not None:
        table.add_row("Cursor X", f"{cursor_pos[0]:+.3f}")
        table.add_row("Cursor Y", f"{cursor_pos[1]:+.3f}")

    if target_positions:
        table.add_row("", "")  # Separator
        for motor_name, pos in target_positions.items():
            table.add_row(f"{motor_name} Target", str(pos))

    if offsets:
        table.add_row("", "")  # Separator
        for motor_name, offset in offsets.items():
            table.add_row(f"{motor_name} Offset", f"{offset:+d}")

    return Panel(table, title="Trackpad Control", border_style="blue")


@app.command()
def trackpad_control(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Control robot with trackpad input."""
    control_with_trackpad(config)


if __name__ == "__main__":
    app()


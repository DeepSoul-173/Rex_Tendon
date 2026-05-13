"""Main CLI entry point for Rex Tendon."""

import logging
import typer
from typing import Optional
from rich.console import Console

from .common.logging import setup_logging

# Initial basic logging setup (will be reconfigured based on CLI args)
setup_logging()

from .hardware.motor_calibration import calibrate as calibrate_command
from .hardware.trackpad_control import trackpad_control as trackpad_command
from .control.primitives import run as primitive_command
from .control.idle import test_idle as idle_command
from .perception.debug_perception import debug_perception as debug_command
from .training.vision.training import app as vision_app
from .training.rl.training import app as rl_app
from .training.rl.generate_mujoco_xml import generate_xml as xml_command
from .training.rl.pick_place_training import app as pick_place_app
# from .training.vision.training import app as vision_app
# from .training.vision.extract_frames import extract_frames as frames_command
# from .training.vision.generate_synthetic_data import generate_data as data_command
# from .training.vision.record_data import app as record_app

app = typer.Typer(
    name="rex-tendon",
    help="A lean tentacle robot controller with stereo vision and RL",
    rich_markup_mode="rich",
)
console = Console()


def setup_logging_callback(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
) -> None:
    """Configure logging based on log level."""
    try:
        level = getattr(logging, log_level.upper())
    except AttributeError:
        console.print(f"[red]Invalid log level: {log_level}[/red]")
        console.print("Valid levels: DEBUG, INFO, WARNING, ERROR")
        raise typer.Exit(1)

    setup_logging(level)


# Add the callback to handle global options
app.callback()(setup_logging_callback)

# Add commands
app.command("calibrate", help="Motor calibration")(calibrate_command)
app.command("primitive", help="Test motion primitive behaviors")(primitive_command)
app.command("idle", help="Test idle motion with breathing pattern")(idle_command)
app.command("trackpad", help="Trackpad control interface")(trackpad_command)
# app.command("extract-frames", help="Extract frames from videos using k-means")(
#     frames_command
# )
# app.command("synthetic-images", help="Generate synthetic training images")(data_command)
app.command("debug-perception", help="Debug stereo vision and triangulation")(
    debug_command
)
app.command("generate-xml", help="Generate MuJoCo XML models")(xml_command)

# Add sub apps
# app.add_typer(vision_app, name="vision", help="Vision training and inference")
app.add_typer(rl_app, name="rl", help="RL training and evaluation")
app.add_typer(pick_place_app, name="pick-place", help="Pick-and-place RL training and evaluation")
# app.add_typer(record_app, name="record", help="Record data from stereo camera")


@app.command()
def hand_control(
    xml: str = typer.Option(
        "rex_assets/rex_simulation/pick_and_place_scene.xml",
        "--xml",
        help="MuJoCo XML scene file",
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model", help="Trained RL model (.zip) for RL mode"
    ),
    camera: int = typer.Option(0, "--camera", help="Webcam device ID"),
    mode: str = typer.Option("direct", "--mode", help="Control mode: direct or rl"),
    smoothing: float = typer.Option(0.3, "--smoothing", help="Action smoothing factor"),
) -> None:
    """Launch hand-controlled tentacle simulation using webcam."""
    from .control.hand_sim_controller import run_hand_controller

    if mode == "rl" and not model_path:
        console.print("[yellow]Warning: RL mode but no --model. Using direct mode.[/yellow]")
        mode = "direct"

    console.print("[bold blue]Launching Hand Control[/bold blue]")
    console.print(f"  Mode: {mode.upper()}")
    console.print(f"  Camera: {camera}")
    console.print(f"  Smoothing: {smoothing}")

    run_hand_controller(
        xml_path=xml,
        model_path=model_path,
        camera_id=camera,
        mode=mode,
        smoothing=smoothing,
    )


@app.command()
def orchestrate(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to orchestrator configuration"
    ),
    hardware_config_path: Optional[str] = typer.Option(
        None, "--hardware-config", help="Path to hardware configuration"
    ),
    perception_config_path: Optional[str] = typer.Option(
        None, "--perception-config", help="Path to perception configuration"
    ),
    control_config_path: Optional[str] = typer.Option(
        None, "--control-config", help="Path to control configuration"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """Start the orchestrator runtime."""
    import asyncio
    from pathlib import Path

    console.print("[bold blue]Starting Orchestrator Application[/bold blue]")

    try:
        from .orchestrator import OrchestratorApp
        from .configs.loaders import (
            get_orchestrator_config,
            get_hardware_config,
            get_perception_config,
            get_control_config,
        )

        # Load configurations
        orchestrator_config = get_orchestrator_config(config)
        hardware_config = get_hardware_config(hardware_config_path)
        perception_config = get_perception_config(perception_config_path)
        control_config = get_control_config(control_config_path)

        # Display config file paths
        config_files = [
            ("Orchestrator", config),
            ("Hardware", hardware_config_path),
            ("Perception", perception_config_path),
            ("Control", control_config_path),
        ]

        for config_type, config_path in config_files:
            if config_path:
                config_file = Path(config_path)
                if config_file.exists():
                    console.print(
                        f"Loading {config_type.lower()} config from: [cyan]{config_path}[/cyan]"
                    )
                else:
                    console.print(
                        f"[yellow]Warning: {config_type} config file not found: {config_path}[/yellow]"
                    )

        # Create orchestrator application
        console.print("[dim]Creating orchestrator application...[/dim]")
        orchestrator_app = OrchestratorApp(
            orchestrator_config=orchestrator_config,
            hardware_config=hardware_config,
            perception_config=perception_config,
            control_config=control_config,
        )
        console.print("[dim]Orchestrator application created successfully[/dim]")

        # Run the async application
        console.print("[green]Launching orchestrator...[/green]")
        asyncio.run(orchestrator_app.start())

    except KeyboardInterrupt:
        console.print("\n[yellow]Application stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error running orchestrator: {e}[/red]")
        if debug:
            import logging

            logging.exception("Full traceback for debugging:")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()


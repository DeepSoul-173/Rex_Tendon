"""Unified vision training using YOLO models with configuration."""

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console

from ...configs.loaders import get_vision_training_config

console = Console()
app = typer.Typer(help="Vision training utilities")


def train_yolo_model(
    dataset_yaml: Path,
    config: Optional[str] = None,
) -> Dict[str, Any]:
    """Train YOLO model using configuration file.

    Args:
        dataset_yaml: Path to dataset configuration file
        config: Optional path to training configuration file

    Returns:
        Training results dictionary
    """
    # Lazy import to avoid loading ultralytics when not needed
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for YOLO training. Install with: pip install ultralytics"
        )

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    # Load configuration
    training_config = get_vision_training_config(config)

    # Initialize and train model
    model = YOLO(training_config.base_model)

    results = model.train(
        data=str(dataset_yaml),
        epochs=training_config.epochs,
        imgsz=training_config.image_size,
        batch=training_config.batch_size,
        device=training_config.device,
        project=training_config.project_name,
        name=training_config.experiment_name,
        **training_config.additional_params,
    )

    # Export to ONNX if configured
    if training_config.export_to_onnx:
        console.print("[blue]Exporting model to ONNX...[/blue]")
        model.export(
            format="onnx",
            imgsz=training_config.image_size,
            optimize=training_config.onnx_optimize,
            dynamic=training_config.onnx_dynamic,
            simplify=training_config.onnx_simplify,
        )

    return results


def evaluate_yolo_model(
    model_path: Path,
    dataset_yaml: Path,
    config: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate YOLO model on validation set."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required for evaluation")

    training_config = get_vision_training_config(config)
    model = YOLO(str(model_path))
    results = model.val(data=str(dataset_yaml), device=training_config.device)
    return results


def predict_yolo_model(
    model_path: Path,
    image_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[str] = None,
    confidence: float = 0.5,
    show: bool = True,
) -> None:
    """Run YOLO model prediction on an image."""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        raise ImportError("ultralytics and opencv-python required for prediction")

    training_config = get_vision_training_config(config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load model and run prediction
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        device=training_config.device,
        save=output_path is not None,
        show=show,
    )

    # Save to specific path if requested
    if output_path and results:
        result = results[0]
        annotated_img = result.plot()
        cv2.imwrite(str(output_path), annotated_img)
        console.print(f"Prediction saved to: {output_path}")


@app.command()
def train(
    dataset_yaml: str = typer.Argument(..., help="Path to dataset YAML file"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Train vision model using YOLO."""
    try:
        console.print("[blue]Starting YOLO training...[/blue]")
        console.print(f"Dataset: {dataset_yaml}")

        results = train_yolo_model(Path(dataset_yaml), config)
        console.print("[green]✓ Training completed successfully![/green]")

    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    dataset_yaml: str = typer.Argument(..., help="Path to dataset YAML file"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Evaluate trained vision model."""
    try:
        console.print(f"[blue]Evaluating model: {model_path}[/blue]")
        console.print(f"Dataset: {dataset_yaml}")

        results = evaluate_yolo_model(Path(model_path), Path(dataset_yaml), config)
        console.print("[green]✓ Evaluation completed![/green]")

    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    image_path: str = typer.Argument(..., help="Path to input image"),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for prediction image"
    ),
    confidence: float = typer.Option(
        0.5, "--confidence", "-c", help="Confidence threshold"
    ),
    show: bool = typer.Option(True, "--show", help="Display result"),
    config: Optional[str] = typer.Option(None, "--config", help="Config file path"),
) -> None:
    """Run vision inference on single image."""
    try:
        console.print(f"[blue]Running prediction on: {image_path}[/blue]")

        output_path_obj = Path(output_path) if output_path else None
        predict_yolo_model(
            Path(model_path),
            Path(image_path),
            output_path_obj,
            config,
            confidence,
            show,
        )

        console.print("[green]✓ Prediction completed![/green]")

    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        raise typer.Exit(1)


"""Perception configuration for stereo vision and detection."""

from pathlib import Path
from typing import Dict
from pydantic import Field
from .base import BaseConfig


class PerceptionConfig(BaseConfig):
    """Configuration for perception components."""

    # Model paths
    yolo_model_path: Path = Field(
        default=Path("rex_assets/rex_models/vision/tentacle_tracking_baseline.onnx"),
        description="Path to YOLO model file (.pt or .onnx)",
    )

    # Camera calibration
    camera_calibration_path: Path = Field(
        default=Path("rex_assets/rex_hardware/calibration"),
        description="Path to camera calibration directory containing stereo_params.pickle",
    )

    # Camera parameters
    camera_index: int = Field(default=0, description="Camera device index")

    stereo_resolution: tuple[int, int] = Field(
        default=(3840, 1520), description="Stereo camera resolution (width, height)"
    )

    # Detection parameters
    confidence_threshold: float = Field(
        default=0.3, description="Confidence threshold for object detection"
    )

    yolo_device: str = Field(
        default="cpu", description="Device for YOLO inference ('cpu', '0', etc.)"
    )

    # Coordinate transformation parameters
    units_to_meters: float = Field(
        default=0.05, description="Scale factor to convert calibration units to meters"
    )

    rotation_angle_deg: float = Field(
        default=35,
        description="Rotation angle for coordinate frame alignment (degrees)",
    )

    y_translation_m: float = Field(
        default=-0.03, description="Y-axis translation offset in meters"
    )

    # Coordinate limits for triangulated points
    coordinate_limits: Dict[str, Dict[str, float]] = Field(
        default={
            "X": {"clip_min": -0.20, "clip_max": 0.20},
            "Y": {"clip_min": 0.0, "clip_max": 0.43},
            "Z": {"clip_min": -0.4, "clip_max": -0.10},
        },
        description="Coordinate limits for clipping triangulated 3D points",
    )

    # Visualization

    dashboard_figure_size: tuple[int, int] = Field(
        default=(6, 6), description="Dashboard figure size (width, height) in inches"
    )


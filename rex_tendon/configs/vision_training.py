"""Vision training configuration for YOLO and computer vision workflows."""

from typing import Dict, Any
from pydantic import Field
from .base import BaseConfig


class VisionTrainingConfig(BaseConfig):
    """Configuration for vision training workflows."""

    # Model configuration
    base_model: str = Field(
        default="yolo11n.pt", description="Base YOLO model to start training from"
    )

    # Training parameters
    epochs: int = Field(default=50, description="Number of training epochs")

    image_size: int = Field(default=640, description="Training image size in pixels")

    batch_size: int = Field(default=16, description="Training batch size")

    device: str = Field(
        default="0", description="Device for training ('cpu', '0', '0,1', etc.)"
    )

    # Project organization
    project_name: str = Field(
        default="yolo_training", description="Project directory name for training runs"
    )

    experiment_name: str = Field(
        default="exp", description="Experiment name within project"
    )

    # Export settings
    export_to_onnx: bool = Field(
        default=True, description="Whether to export trained model to ONNX"
    )

    onnx_optimize: bool = Field(
        default=True, description="Whether to optimize ONNX export"
    )

    onnx_dynamic: bool = Field(
        default=True, description="Whether to use dynamic shapes in ONNX export"
    )

    onnx_simplify: bool = Field(
        default=True, description="Whether to simplify ONNX model"
    )

    # Additional training parameters
    additional_params: Dict[str, Any] = Field(
        default={
            "patience": 30,
            "save_period": 10,
            "workers": 8,
            "optimizer": "auto",
            "lr0": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
        },
        description="Additional parameters passed to YOLO training",
    )


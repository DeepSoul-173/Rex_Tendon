"""Configuration module for Rex Tendon."""

from .loaders import (
    get_hardware_config,
    get_perception_config,
    get_vision_training_config,
    get_rl_training_config,
    get_orchestrator_config,
    get_control_config,
)

__all__ = [
    "get_hardware_config",
    "get_perception_config",
    "get_vision_training_config",
    "get_rl_training_config",
    "get_orchestrator_config",
    "get_control_config",
]


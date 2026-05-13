"""Configuration loading utilities."""

from typing import Optional

from .hardware import HardwareConfig
from .orchestrator import OrchestratorConfig
from .perception import PerceptionConfig
from .vision_training import VisionTrainingConfig
from .rl_training import RLTrainingConfig
from .control import ControlConfig


def get_hardware_config(config_file: Optional[str] = None) -> HardwareConfig:
    """Get hardware configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        HardwareConfig instance
    """
    return HardwareConfig.load(config_file)


def get_perception_config(config_file: Optional[str] = None) -> PerceptionConfig:
    """Get perception configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        PerceptionConfig instance
    """
    return PerceptionConfig.load(config_file)


def get_vision_training_config(
    config_file: Optional[str] = None,
) -> VisionTrainingConfig:
    """Get vision training configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        VisionTrainingConfig instance
    """
    return VisionTrainingConfig.load(config_file)


def get_rl_training_config(config_file: Optional[str] = None) -> RLTrainingConfig:
    """Get RL training configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        RLTrainingConfig instance
    """
    return RLTrainingConfig.load(config_file)


def get_orchestrator_config(config_file: Optional[str] = None) -> OrchestratorConfig:
    """Get orchestrator configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        OrchestratorConfig instance
    """
    return OrchestratorConfig.load(config_file)


def get_control_config(config_file: Optional[str] = None) -> ControlConfig:
    """Get control configuration with optional YAML override.

    Args:
        config_file: Optional path to YAML config file

    Returns:
        ControlConfig instance
    """
    return ControlConfig.load(config_file)


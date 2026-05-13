import re
import yaml
from pathlib import Path
from typing import TypeVar, Type, Optional, Dict, Any
from pydantic_settings import BaseSettings

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseSettings):
    """Base configuration class with common functionality."""

    @classmethod
    def _camel_to_snake(cls, name: str) -> str:
        """Convert CamelCase to snake_case."""
        # Insert an underscore before any uppercase letter that follows a lowercase letter
        s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return s1.lower()

    @classmethod
    def load(cls: Type[T], config_file: Optional[str] = None) -> T:
        """Unified config loading method.

        Args:
            config_file: Optional path to YAML config file

        Returns:
            Instance of the configuration class (with defaults if no file provided)
        """
        if config_file:
            if not Path(config_file).exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}

            class_name = cls.__name__.replace("Config", "")
            section_name = cls._camel_to_snake(class_name)
            section_data = data.get(section_name, {})

            return cls(**section_data)
        return cls()


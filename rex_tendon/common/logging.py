"""Simple centralized logging setup."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the application.

    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


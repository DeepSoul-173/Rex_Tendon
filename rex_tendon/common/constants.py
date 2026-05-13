"""Constants for motor geometry and system parameters."""

import numpy as np
from typing import Dict

MOTOR_NAMES = ["1", "2", "3"]
MOTOR_ONE_FULL_TURN_TICKS = 4096
MOTOR_POSITION_MIN = -32768
MOTOR_POSITION_MAX = 32767

MOTOR_NORMALIZED_POSITIONS: Dict[str, np.ndarray] = {
    "1": np.array([0.0, 1.0]),
    "2": np.array([0.866, -0.5]),
    "3": np.array([-0.866, -0.5]),
}


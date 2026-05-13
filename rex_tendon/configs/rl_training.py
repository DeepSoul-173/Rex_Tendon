"""RL training configuration for reinforcement learning workflows."""

from typing import Dict, List, Optional, Tuple
from pydantic import Field
from .base import BaseConfig


class RLEnvironmentConfig(BaseConfig):
    """Configuration for RL environment parameters."""

    # MuJoCo simulation settings
    xml_file: str = Field(
        default="rex_assets/rex_simulation/tentacle.xml",
        description="Path to MuJoCo XML file (relative to rl module or absolute)",
    )
    simulation_length_seconds: float = Field(
        default=8.0, description="Length of simulation in seconds"
    )
    time_between_steps_seconds: float = Field(
        default=0.08, description="Time between environment steps"
    )

    # Actuator/tendon settings
    initial_actuator_position: Tuple[float, float] = Field(
        default=(0.23, 0.23),
        description="Initial tendon length(s) - float or (min, max) for randomization",
    )

    # Reward shaping
    reward_distance_scale: float = Field(
        default=1.0, description="Scale for distance-based reward"
    )
    distance_penalty_exponent: float = Field(
        default=1.0, description="Exponent for distance penalty (1=linear, 2=squared)"
    )
    action_change_penalty_scale: float = Field(
        default=0.25, description="Scale for action change penalty"
    )

    # Environment objects
    tip_site_name: str = Field(default="tip_center", description="Name of tip site")
    target_bounds_min: List[float] = Field(
        default=[-0.2, -0.15, 0.12], description="Minimum target position bounds"
    )
    target_bounds_max: List[float] = Field(
        default=[-0.1, 0.15, 0.22], description="Maximum target position bounds"
    )

    # Observation settings
    num_frames: int = Field(
        default=4, description="Number of observation frames to stack"
    )
    include_actuator_lengths_in_obs: bool = Field(
        default=True, description="Include actuator lengths in observation"
    )

    # Action space
    max_2d_action_magnitude: float = Field(
        default=1.0, description="Maximum magnitude for 2D actions"
    )

    # Trajectory generation
    pause_probability: float = Field(
        default=0.2, description="Probability of trajectory pause segments"
    )
    min_pause_duration: float = Field(
        default=0.5, description="Minimum pause duration in seconds"
    )
    max_pause_duration: float = Field(
        default=3.0, description="Maximum pause duration in seconds"
    )
    min_move_duration: float = Field(
        default=0.5, description="Minimum move duration in seconds"
    )
    max_move_duration: float = Field(
        default=2.0, description="Maximum move duration in seconds"
    )

    # Sim2Real domain randomization
    randomize_dynamics: bool = Field(
        default=True, description="Enable dynamics randomization"
    )
    randomization_factors: Dict[str, float] = Field(
        default={
            "body_mass": 0.05,
            "body_inertia": 0.05,
            "dof_damping": 0.05,
            "jnt_stiffness": 0.05,
            "geom_friction": 0.05,
            "geom_solref": 0.05,
            "geom_solimp": 0.05,
            "actuator_gainprm": 0.0,
            "actuator_biasprm": 0.0,
        },
        description="Randomization factors for physics parameters",
    )

    # Observation noise
    add_observation_noise: bool = Field(
        default=True, description="Add Gaussian noise to observations"
    )
    observation_noise_scale: float = Field(
        default=0.005, description="Standard deviation of observation noise"
    )

    # Fixed target (optional)
    fixed_target_position: Optional[Tuple[float, float, float]] = Field(
        default=None, description="Fixed target position (overrides trajectory)"
    )


class RLTrainingParamsConfig(BaseConfig):
    """Configuration for RL training parameters."""

    # Environment setup
    num_envs: int = Field(default=6, description="Number of parallel environments")

    # Training duration
    total_timesteps: int = Field(
        default=100_000_000, description="Total training timesteps"
    )

    # Evaluation
    eval_freq: int = Field(default=10000, description="Timesteps between evaluations")
    n_eval_episodes: int = Field(
        default=5, description="Number of episodes per evaluation"
    )

    # Checkpointing
    save_freq: int = Field(default=50000, description="Timesteps between model saves")
    log_dir_base: str = Field(default="results", description="Base directory for logs")

    # PPO hyperparameters
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    n_steps: int = Field(
        default=400, description="Steps collected per env before update"
    )
    batch_size: int = Field(default=64, description="Training batch size")
    n_epochs: int = Field(default=5, description="Optimization epochs per update")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    clip_range: float = Field(default=0.2, description="PPO clip range")
    ent_coef: float = Field(default=0.0, description="Entropy coefficient")

    # Network architecture
    net_arch: str = Field(
        default="256-256",
        description="Network architecture ",
    )
    activation_fn: str = Field(
        default="Tanh", description="Activation function name (e.g., 'Tanh', 'ReLU')"
    )


class RLEvaluationConfig(BaseConfig):
    """Configuration for RL evaluation parameters."""

    num_episodes: int = Field(default=10, description="Number of episodes to evaluate")
    render_delay: float = Field(
        default=0.05, description="Delay between rendered frames"
    )
    deterministic_actions: bool = Field(
        default=True, description="Use deterministic policy actions"
    )
    render_mode: Optional[str] = Field(
        default=None,
        description="Render mode ('human', 'rgb_array', None for no rendering)",
    )


class RLTrainingConfig(BaseConfig):
    """Combined RL training configuration."""

    rl_env: RLEnvironmentConfig = Field(
        default_factory=RLEnvironmentConfig, description="RL environment config"
    )
    rl_training_params: RLTrainingParamsConfig = Field(
        default_factory=RLTrainingParamsConfig, description="RL training config"
    )
    rl_evaluation: RLEvaluationConfig = Field(
        default_factory=RLEvaluationConfig, description="RL evaluation config"
    )


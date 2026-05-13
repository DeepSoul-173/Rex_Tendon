"""Pick-and-place RL configuration models."""

from typing import Dict, List, Optional, Tuple
from pydantic import Field
from .base import BaseConfig


class PickPlaceEnvConfig(BaseConfig):
    """Configuration for the pick-and-place RL environment."""

    # MuJoCo simulation settings
    xml_file: str = Field(
        default="rex_assets/rex_simulation/pick_and_place_scene.xml",
        description="Path to MuJoCo XML file",
    )
    simulation_length_seconds: float = Field(
        default=12.0, description="Length of simulation in seconds"
    )
    time_between_steps_seconds: float = Field(
        default=0.08, description="Time between environment steps"
    )

    # Actuator/tendon settings
    initial_actuator_position: Tuple[float, float] = Field(
        default=(0.20, 0.26),
        description="Initial tendon length range (min, max) for randomization",
    )

    # Environment objects
    tip_site_name: str = Field(default="tip_center", description="Name of tip site")
    place_zone_site_name: str = Field(
        default="place_zone", description="Name of place zone site"
    )

    # Object names (must match XML)
    object_names: List[str] = Field(
        default=["obj_cube", "obj_cylinder", "obj_sphere"],
        description="Names of graspable object bodies in XML",
    )
    object_site_names: List[str] = Field(
        default=["obj_cube_site", "obj_cylinder_site", "obj_sphere_site"],
        description="Names of object site elements for position tracking",
    )
    object_geom_names: List[str] = Field(
        default=["obj_cube_geom", "obj_cylinder_geom", "obj_sphere_geom"],
        description="Names of object geom elements",
    )
    grasp_constraint_names: List[str] = Field(
        default=["grasp_cube", "grasp_cylinder", "grasp_sphere"],
        description="Names of weld equality constraints for grasping",
    )

    # Grasping parameters
    grasp_distance_threshold: float = Field(
        default=0.06, description="Distance threshold for grasping (meters)"
    )
    grasp_consecutive_steps: int = Field(
        default=3, description="Consecutive steps within threshold to trigger grasp"
    )
    place_distance_threshold: float = Field(
        default=0.06, description="Distance threshold for successful placement (meters)"
    )

    # Reward shaping
    reach_reward_scale: float = Field(
        default=1.0, description="Scale for reach distance reward"
    )
    grasp_bonus: float = Field(
        default=5.0, description="Bonus reward for successful grasp"
    )
    transport_reward_scale: float = Field(
        default=2.0, description="Scale for transport distance reward"
    )
    place_bonus: float = Field(
        default=10.0, description="Bonus reward for successful placement"
    )
    drop_penalty: float = Field(
        default=3.0, description="Penalty for dropping object"
    )
    action_change_penalty_scale: float = Field(
        default=0.1, description="Scale for action change penalty"
    )
    action_jerk_penalty_scale: float = Field(
        default=0.0, description="Scale for second-order action change penalty"
    )
    object_progress_reward_scale: float = Field(
        default=0.0,
        description="Reward scale for object progress toward the place zone",
    )
    stable_contact_bonus_scale: float = Field(
        default=0.0,
        description="Small bonus scale for maintaining close tip/object contact",
    )
    assisted_grasp_enabled: bool = Field(
        default=False,
        description="Kinematically carries object after grasp for curriculum training",
    )
    assisted_grasp_follow_alpha: float = Field(
        default=1.0,
        description="Object follow blend used by assisted grasp curriculum",
    )

    # Curriculum learning
    curriculum_enabled: bool = Field(
        default=True, description="Enable curriculum learning"
    )
    reach_only_steps: int = Field(
        default=500_000, description="Steps for reach-only phase"
    )
    reach_grasp_steps: int = Field(
        default=2_000_000, description="Steps for reach+grasp phase (cumulative)"
    )

    # Observation settings
    num_frames: int = Field(
        default=4, description="Number of observation frames to stack"
    )
    include_actuator_lengths_in_obs: bool = Field(
        default=True, description="Include actuator lengths in observation"
    )
    include_relative_observations: bool = Field(
        default=False,
        description="Include tip-object and object-place vectors in observation",
    )
    include_object_velocity_in_obs: bool = Field(
        default=False,
        description="Include active object linear velocity estimate in observation",
    )

    # Object spawn bounds (where objects can be randomly placed on desk)
    object_spawn_bounds_min: List[float] = Field(
        default=[-0.1, -0.1, -0.025],
        description="Minimum XYZ bounds for object spawning",
    )
    object_spawn_bounds_max: List[float] = Field(
        default=[0.1, 0.1, -0.025],
        description="Maximum XYZ bounds for object spawning",
    )

    # Action space
    max_2d_action_magnitude: float = Field(
        default=1.0, description="Maximum magnitude for 2D actions"
    )
    action_smoothing_alpha: float = Field(
        default=1.0,
        description="Blend factor for action smoothing; 1.0 applies raw action",
    )
    max_action_delta_per_step: Optional[float] = Field(
        default=None,
        description="Optional max L2 action change per environment step",
    )

    # Domain randomization
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


class PickPlaceTaskConfig(BaseConfig):
    """Optional predefined pick-and-place task."""

    name: str = Field(default="task1", description="Task name")
    active_object_index: Optional[int] = Field(
        default=None, description="Object index to pick; random if unset"
    )
    object_position: Optional[Tuple[float, float, float]] = Field(
        default=None, description="Fixed XYZ pickup position for active object"
    )
    place_position: Optional[Tuple[float, float, float]] = Field(
        default=None, description="Fixed XYZ place-zone position"
    )
    randomize_object_position: bool = Field(
        default=True, description="Randomize object positions on reset"
    )


class PickPlaceTrainingConfig(BaseConfig):
    """Configuration for pick-and-place RL training parameters."""

    # Environment setup
    num_envs: int = Field(default=4, description="Number of parallel environments")

    # Training duration
    total_timesteps: int = Field(
        default=10_000_000, description="Total training timesteps"
    )

    # Evaluation
    eval_freq: int = Field(default=20000, description="Timesteps between evaluations")
    n_eval_episodes: int = Field(
        default=5, description="Number of episodes per evaluation"
    )

    # Checkpointing
    save_freq: int = Field(default=50000, description="Timesteps between model saves")
    log_dir_base: str = Field(default="rex_results/pick_place", description="Base directory for logs")

    # PPO hyperparameters
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    n_steps: int = Field(default=512, description="Steps collected per env before update")
    batch_size: int = Field(default=128, description="Training batch size")
    n_epochs: int = Field(default=10, description="Optimization epochs per update")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter")
    clip_range: float = Field(default=0.2, description="PPO clip range")
    ent_coef: float = Field(default=0.01, description="Entropy coefficient")

    # Network architecture
    net_arch: str = Field(default="256-256", description="Network architecture")
    activation_fn: str = Field(default="Tanh", description="Activation function name")


class PickPlaceEvaluationConfig(BaseConfig):
    """Configuration for pick-and-place RL evaluation."""

    num_episodes: int = Field(default=10, description="Number of episodes to evaluate")
    render_delay: float = Field(default=0.05, description="Delay between rendered frames")
    deterministic_actions: bool = Field(default=True, description="Use deterministic policy")
    render_mode: Optional[str] = Field(default=None, description="Render mode")


class PickPlaceConfig(BaseConfig):
    """Combined pick-and-place configuration."""

    env: PickPlaceEnvConfig = Field(
        default_factory=PickPlaceEnvConfig, description="Environment config"
    )
    active_task: Optional[str] = Field(
        default=None, description="Name of predefined task to use"
    )
    tasks: Dict[str, PickPlaceTaskConfig] = Field(
        default_factory=dict, description="Predefined pick-and-place tasks"
    )
    training: PickPlaceTrainingConfig = Field(
        default_factory=PickPlaceTrainingConfig, description="Training config"
    )
    evaluation: PickPlaceEvaluationConfig = Field(
        default_factory=PickPlaceEvaluationConfig, description="Evaluation config"
    )


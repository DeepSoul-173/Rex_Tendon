import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")


def test_target_following_action_penalty_uses_previous_tendon_target():
    from rex_tendon.configs.rl_training import RLEnvironmentConfig
    from rex_tendon.training.rl.environment import TentacleTargetFollowingEnv

    env = TentacleTargetFollowingEnv(
        RLEnvironmentConfig(
            simulation_length_seconds=1.0,
            randomize_dynamics=False,
            add_observation_noise=False,
            action_change_penalty_scale=1.0,
        )
    )
    try:
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info["action_penalty"] > 0.0
        assert info["distance"] == info["distance_to_target"]
    finally:
        env.close()


def test_pick_place_enhanced_observation_shape_and_task_config():
    from rex_tendon.configs.pick_place_config import (
        PickPlaceEnvConfig,
        PickPlaceTaskConfig,
    )
    from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv

    config = PickPlaceEnvConfig(
        simulation_length_seconds=1.0,
        randomize_dynamics=False,
        add_observation_noise=False,
        curriculum_enabled=False,
        include_relative_observations=True,
        include_object_velocity_in_obs=True,
    )
    task = PickPlaceTaskConfig(
        active_object_index=0,
        object_position=(0.08, -0.05, 0.0),
        place_position=(-0.08, 0.08, -0.025),
        randomize_object_position=False,
    )

    env = TentaclePickPlaceEnv(config=config, task_config=task)
    try:
        obs, _ = env.reset(seed=1)
        assert env.single_frame_dim == 24
        assert obs.shape == (config.num_frames * 24,)
        expected_position = np.array(task.object_position)
        expected_position[2] = env.object_rest_z[0]
        np.testing.assert_allclose(
            env._get_object_position(0),
            expected_position,
            atol=1e-6,
        )
    finally:
        env.close()


def test_pick_place_success_requires_actual_object_at_place_zone():
    from rex_tendon.configs.pick_place_config import PickPlaceEnvConfig
    from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv

    config = PickPlaceEnvConfig(
        simulation_length_seconds=1.0,
        randomize_dynamics=False,
        add_observation_noise=False,
        curriculum_enabled=False,
        assisted_grasp_enabled=False,
    )
    env = TentaclePickPlaceEnv(config=config)
    try:
        env.reset(seed=1)
        env.is_grasped = True
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated
        assert not info["place_success"]
        assert not info["tip_object_contact"]
        assert info["object_to_place_distance"] > config.place_distance_threshold
    finally:
        env.close()


def test_pick_place_proximity_triggers_grasp_in_hybrid_mode():
    """Hybrid grasp: being within the proximity window latches a grasp even
    without a measured contact force (contact OR proximity)."""
    from rex_tendon.configs.pick_place_config import (
        PickPlaceEnvConfig,
        PickPlaceTaskConfig,
    )
    from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv

    config = PickPlaceEnvConfig(
        simulation_length_seconds=1.0,
        randomize_dynamics=False,
        add_observation_noise=False,
        curriculum_enabled=False,
        assisted_grasp_enabled=False,
        grasp_distance_threshold=1.0,
        grasp_consecutive_steps=1,
    )
    task = PickPlaceTaskConfig(
        active_object_index=0,
        object_position=(0.16, 0.16, 0.0),
        randomize_object_position=False,
    )
    env = TentaclePickPlaceEnv(config=config, task_config=task)
    try:
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["tip_to_object_distance"] < config.grasp_distance_threshold
        assert info["grasp_within_proximity"]
        assert info["is_grasped"]
    finally:
        env.close()


def test_pick_place_far_object_without_contact_does_not_grasp():
    """Outside both the (tiny) proximity window and with no real contact, the
    grasp detector must stay False — guards against false-positive latching."""
    from rex_tendon.configs.pick_place_config import (
        PickPlaceEnvConfig,
        PickPlaceTaskConfig,
    )
    from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv

    config = PickPlaceEnvConfig(
        simulation_length_seconds=1.0,
        randomize_dynamics=False,
        add_observation_noise=False,
        curriculum_enabled=False,
        assisted_grasp_enabled=False,
        grasp_distance_threshold=0.01,
        grasp_consecutive_steps=1,
    )
    task = PickPlaceTaskConfig(
        active_object_index=0,
        object_position=(0.12, 0.0, 0.0),
        randomize_object_position=False,
    )
    env = TentaclePickPlaceEnv(config=config, task_config=task)
    try:
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["tip_to_object_distance"] > config.grasp_distance_threshold
        assert not info["tip_object_contact"]
        assert not info["is_grasped"]
    finally:
        env.close()

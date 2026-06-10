"""Regression tests for the carry-phase reward and stacking-metric fixes.

Covers the defects diagnosed on the 2026-06-10 stacking run:
  - no gradient toward the place target once grasped (carry_reach term)
  - lift measured from table-rest height, so stacked cubes count as
    "lifted" at spawn
  - occlusion proxy always-on for stacking (stack-mates share an xy)
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from types import SimpleNamespace


def _stacking_env(carry_scale: float = 0.0):
    from rex_tendon.configs.pick_place_config import (
        PickPlaceEnvConfig,
        PickPlaceTaskConfig,
    )
    from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv

    config = PickPlaceEnvConfig(
        simulation_length_seconds=2.0,
        randomize_dynamics=False,
        add_observation_noise=False,
        carry_reach_reward_scale=carry_scale,
    )
    task = PickPlaceTaskConfig(
        stacking=True,
        stack_count=2,
        source_xy=(0.08, 0.0),
        target_xy=(-0.08, 0.0),
    )
    # Huge global step => FULL curriculum phase (grasp + place enabled).
    return TentaclePickPlaceEnv(
        config=config,
        task_config=task,
        global_step_counter=SimpleNamespace(value=10**9),
    )


def test_carry_reach_scale_defaults_to_zero():
    from rex_tendon.configs.pick_place_config import PickPlaceEnvConfig

    assert PickPlaceEnvConfig().carry_reach_reward_scale == 0.0


def test_stacking_occlusion_not_flagged():
    env = _stacking_env()
    try:
        env.reset(seed=3)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["occluded"] is False
    finally:
        env.close()


def test_stacked_top_cube_not_lifted_at_spawn():
    env = _stacking_env()
    try:
        env.reset(seed=3)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # Old behaviour: top of a 2-stack spawned 0.02 m above table rest and
        # immediately counted as lifted. Now lift is measured from spawn.
        assert info["object_lift_height"] < 0.005
        assert not info["object_lifted"]
    finally:
        env.close()


def test_carry_reach_term_applies_only_when_grasped():
    envs = {k: _stacking_env(carry_scale=k) for k in (0.0, 1.0)}
    try:
        rewards, infos = {}, {}
        for k, env in envs.items():
            env.reset(seed=7)
            env._activate_grasp(env.active_object_idx)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, rewards[k], _, _, infos[k] = env.step(action)

        # Identical seeds/actions => identical physics; the reward difference
        # must be exactly the carry term: scale * tip_to_place.
        assert infos[1.0]["is_grasped"]
        tip_to_place = infos[1.0]["tip_to_place_distance"]
        assert tip_to_place > 0.05  # tip starts nowhere near the target slot
        assert rewards[0.0] - rewards[1.0] == pytest.approx(
            tip_to_place, abs=1e-6
        )
    finally:
        for env in envs.values():
            env.close()


def test_carry_reach_term_absent_when_not_grasped():
    envs = {k: _stacking_env(carry_scale=k) for k in (0.0, 1.0)}
    try:
        rewards = {}
        for k, env in envs.items():
            env.reset(seed=7)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, rewards[k], _, _, info = env.step(action)
            assert not info["is_grasped"]
        assert rewards[0.0] == pytest.approx(rewards[1.0], abs=1e-9)
    finally:
        for env in envs.values():
            env.close()


def test_episode_ever_grasped_reported():
    env = _stacking_env()
    try:
        env.reset(seed=3)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["episode_ever_grasped"] is False
        env._activate_grasp(env.active_object_idx)
        _, _, _, _, info = env.step(action)
        assert info["episode_ever_grasped"] is True
    finally:
        env.close()


def test_stack_config_loads_with_fix():
    from rex_tendon.training.rl.pick_place_training import load_pick_place_config

    cfg = load_pick_place_config("rex_tendon/configs/pick_place_stack.yaml")
    assert cfg.env.carry_reach_reward_scale == 1.0
    assert cfg.training.ent_coef == 0.0075

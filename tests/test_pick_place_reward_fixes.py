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


def _stacking_env(carry_scale: float = 0.0, **env_overrides):
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
        **env_overrides,
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
    assert cfg.env.grasp_requires_contact is True
    assert cfg.env.place_mode == "release"
    assert cfg.env.place_distance_threshold == 0.03


def _float_cube_below_tip(env, gap: float = 0.025):
    """Zero gravity and park the active cube `gap` m below the tip:
    inside grasp proximity (0.035) but without any contact."""
    env.model.opt.gravity[:] = 0.0
    tip = env._get_tip_position()
    env._set_object_position(
        env.active_object_idx, tip + np.array([0.0, 0.0, -gap])
    )


def test_proximity_alone_grasps_only_in_legacy_mode():
    action = None
    for requires_contact, should_grasp in ((False, True), (True, False)):
        env = _stacking_env(grasp_requires_contact=requires_contact)
        try:
            env.reset(seed=11)
            _float_cube_below_tip(env)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            grasped = False
            for _ in range(4):  # > grasp_consecutive_steps
                _float_cube_below_tip(env)  # keep it parked despite drift
                _, _, _, _, info = env.step(action)
                grasped = grasped or info["is_grasped"]
            assert grasped == should_grasp, (
                f"requires_contact={requires_contact}: expected grasped="
                f"{should_grasp}, got {grasped}"
            )
        finally:
            env.close()


def test_release_mode_releases_instead_of_snapping():
    # When a carried cube comes within the threshold of the place zone, the
    # grasp must open and a settle window must start — with NO teleport.
    env = _stacking_env(
        grasp_requires_contact=True, place_mode="release", place_settle_steps=3
    )
    try:
        env.reset(seed=11)
        env.model.opt.gravity[:] = 0.0  # keep the parked cube where it is
        env._activate_grasp(env.active_object_idx)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(action)
        cube_pos = env._get_object_position(env.active_object_idx).copy()
        env._set_place_zone_position(cube_pos)  # trigger on next step

        _, _, _, _, info = env.step(action)
        assert info["place_settling"] is True
        assert info["is_grasped"] is False  # grasp opened...
        assert info["num_placed"] == 0  # ...but nothing counted yet
        # And the cube was not teleported anywhere (residual release velocity
        # may drift it a little in zero-g, hence the loose tolerance).
        np.testing.assert_allclose(
            env._get_object_position(env.active_object_idx), cube_pos, atol=5e-2
        )
    finally:
        env.close()


def test_settle_judgment_counts_cube_resting_on_slot():
    env = _stacking_env(
        place_mode="release", place_settle_steps=3, place_distance_threshold=0.03
    )
    try:
        env.reset(seed=11)
        idx = env.active_object_idx
        slot0 = env._stacking_slot_position(0)
        # Park the cube on the table 1 cm from the canonical slot and
        # manufacture the settling state: the judgment must accept it.
        env._set_object_position(idx, slot0 + np.array([0.0, 0.01, 0.001]))
        env._place_pending_idx = idx
        env._place_settle_countdown = 1
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["cubes_placed_this_step"] is True
        assert info["num_placed"] == 1
        assert info["place_failed_this_step"] is False
    finally:
        env.close()


def test_settle_judgment_rejects_cube_far_from_slot():
    env = _stacking_env(
        place_mode="release", place_settle_steps=3, place_distance_threshold=0.03
    )
    try:
        env.reset(seed=11)
        idx = env.active_object_idx
        slot0 = env._stacking_slot_position(0)
        off_slot = slot0 + np.array([0.0, 0.06, 0.001])  # 6 cm off target
        env._set_object_position(idx, off_slot)
        env._place_pending_idx = idx
        env._place_settle_countdown = 1
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["place_failed_this_step"] is True
        assert info["num_placed"] == 0
        # Failure must not teleport the cube to the slot either.
        final_pos = env._get_object_position(idx)
        assert float(np.linalg.norm(final_pos - slot0)) > 0.03
    finally:
        env.close()

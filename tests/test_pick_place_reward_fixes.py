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


def test_grasp_hold_bonus_applies_only_when_grasped():
    # Grasped: reward difference between hold_bonus 0.15 and 0 is exactly 0.15.
    rewards = {}
    for bonus in (0.0, 0.15):
        env = _stacking_env(grasp_hold_bonus=bonus)
        try:
            env.reset(seed=7)
            env._activate_grasp(env.active_object_idx)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, rewards[bonus], _, _, info = env.step(action)
            assert info["is_grasped"]
        finally:
            env.close()
    assert rewards[0.15] - rewards[0.0] == pytest.approx(0.15, abs=1e-9)

    # Not grasped: the bonus must not leak in.
    rewards = {}
    for bonus in (0.0, 0.15):
        env = _stacking_env(grasp_hold_bonus=bonus)
        try:
            env.reset(seed=7)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, rewards[bonus], _, _, info = env.step(action)
            assert not info["is_grasped"]
        finally:
            env.close()
    assert rewards[0.15] == pytest.approx(rewards[0.0], abs=1e-9)


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
    # Run-6 design: budgeted hold bonus + once-per-episode grasp bonus
    # (anti-farming) + per-episode randomized reachable targets
    # (goal-conditioned discoverability).
    assert cfg.env.carry_reach_reward_scale == 0.0
    assert cfg.env.grasp_hold_bonus == 0.15
    assert cfg.env.grasp_hold_bonus_budget_steps == 60
    assert cfg.env.grasp_bonus_first_only is True
    assert cfg.env.stack_target_randomize is True
    assert cfg.env.grasp_proximity_bonus_scale == 0.1
    assert cfg.env.object_progress_reward_scale == 12.0
    assert cfg.training.ent_coef == 0.005
    assert cfg.env.grasp_requires_contact is False
    assert cfg.env.place_mode == "release"
    # Run-7: tolerance matched to the demonstrated ~5 cm carry precision.
    assert cfg.env.place_distance_threshold == 0.045


def test_hold_bonus_budget_expires():
    # With budget=3, the hold bonus must pay for exactly 3 held steps: the
    # reward difference between bonus 0.15 and 0 vanishes from step 4 on.
    rewards = {0.0: [], 0.15: []}
    for bonus in rewards:
        env = _stacking_env(
            grasp_hold_bonus=bonus, grasp_hold_bonus_budget_steps=3
        )
        try:
            env.reset(seed=7)
            env._activate_grasp(env.active_object_idx)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            for _ in range(5):
                _, r, _, _, info = env.step(action)
                assert info["is_grasped"]
                rewards[bonus].append(r)
        finally:
            env.close()
    diffs = [b - a for a, b in zip(rewards[0.0], rewards[0.15])]
    assert diffs[0] == pytest.approx(0.15, abs=1e-9)
    assert diffs[2] == pytest.approx(0.15, abs=1e-9)
    assert diffs[3] == pytest.approx(0.0, abs=1e-9)  # budget exhausted
    assert diffs[4] == pytest.approx(0.0, abs=1e-9)


def _episode_rewards(first_only: bool) -> list[float]:
    """Latch via the env's own trigger, force a release at step 15 (the
    trigger re-latches within that same step — grasp_proximity_count is not
    reset on release), and return the per-step rewards."""
    env = _stacking_env(grasp_bonus_first_only=first_only)
    try:
        env.reset(seed=11)
        env.model.opt.gravity[:] = 0.0
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        rewards = []
        for k in range(20):
            if not env.is_grasped:
                _float_cube_below_tip(env)  # in proximity, trigger latches
            if k == 15:
                env._deactivate_grasp(env.active_object_idx)
            _, r, _, _, info = env.step(action)
            rewards.append(float(r))
        assert info["is_grasped"]  # re-latched and held to the end
        return rewards
    finally:
        env.close()


def test_grasp_bonus_paid_once_per_episode():
    legacy = _episode_rewards(first_only=False)
    gated = _episode_rewards(first_only=True)
    # Identical physics: only the bonus gating differs.
    # First latch (step 2) pays in both designs...
    assert legacy[2] == pytest.approx(gated[2], abs=1e-9)
    assert legacy[2] > 4.0  # the +5 grasp bonus is in there
    # ...the re-latch (step 15) pays only in legacy.
    assert legacy[15] - gated[15] == pytest.approx(5.0, abs=1e-9)


def test_randomized_stack_target_is_reachable_and_clear_of_source():
    env = _stacking_env(stack_target_randomize=True)
    try:
        targets = []
        for seed in range(6):
            env.reset(seed=seed)
            t = env._stack_target_xy.copy()
            targets.append(t.copy())
            r = float(np.linalg.norm(t))
            assert 0.05 - 1e-6 <= r <= 0.10 + 1e-6
            assert float(np.linalg.norm(t - env._stack_source_xy)) >= 0.04
        # Targets actually vary across episodes.
        spread = np.std(np.array(targets), axis=0).sum()
        assert spread > 0.01
    finally:
        env.close()


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


def test_release_attempt_not_regrasped_with_hybrid_trigger():
    """With proximity grasping the tip still hovers inside the grasp radius
    right after a place release; the settle window must suppress re-latching
    so the judgment can complete. Run-4 defect: every attempt was re-grasped
    within 2 steps and withdrawn — 50% grasp rate, 0% place at 2.8M steps."""
    env = _stacking_env(
        grasp_requires_contact=False,  # hybrid trigger, as in the run-4 config
        place_mode="release",
        place_settle_steps=4,
        place_distance_threshold=0.03,
    )
    try:
        env.reset(seed=11)
        env.model.opt.gravity[:] = 0.0  # keep the released cube parked
        env._activate_grasp(env.active_object_idx)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(action)  # assist parks the cube below the tip
        cube_pos = env._get_object_position(env.active_object_idx).copy()
        env._set_place_zone_position(cube_pos)  # trigger release next step

        _, _, _, _, info = env.step(action)
        assert info["place_settling"] is True
        assert info["is_grasped"] is False

        judged = False
        for _ in range(5):
            _, _, _, _, info = env.step(action)
            if info["place_settling"]:
                # No re-latch while the attempt is being judged.
                assert info["is_grasped"] is False
            if info["cubes_placed_this_step"] or info["place_failed_this_step"]:
                judged = True
                break
        assert judged  # the attempt ran to judgment instead of being withdrawn
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

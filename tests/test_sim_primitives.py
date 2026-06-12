"""End-to-end tests for the scripted sim primitives (control/sim_primitives.py).

These run real MuJoCo physics headless: they prove the voice-command backend
can genuinely reach, grasp, carry, and place in the pick-and-place scene.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")


@pytest.fixture()
def arm():
    from rex_tendon.control.sim_primitives import SimArm

    arm = SimArm(viewer=False)
    yield arm
    arm.close()


def test_move_to_reachable_point(arm):
    # Inside the measured SETTLED workspace (coil branch, r up to ~0.07 with
    # baseline assist; dynamic/swinging control reaches farther but scripted
    # positioning works from settled poses).
    target = np.array([0.07, 0.0, 0.045])
    assert arm.move_to(target, tol=0.02)
    assert np.linalg.norm(arm.tip_position()[:2] - target[:2]) <= 0.02


def test_grasp_by_color(arm):
    red = arm.find_object(color="red")
    assert red is not None and red.name == "obj_cube"
    assert arm.grasp(red)
    assert arm.grasped is red
    # Carried object tracks the tip.
    arm.step(20)
    tip_to_obj = np.linalg.norm(arm.tip_position() - red.position(arm.data))
    assert tip_to_obj < 0.03


def test_pick_and_place_at_zone(arm):
    red = arm.find_object(color="red")
    assert arm.grasp(red)
    zone = arm.place_zone_position()
    assert arm.place_at(zone)
    assert arm.grasped is None
    final = red.position(arm.data)
    # The zone is a 6 cm-radius disc at the edge of the settled workspace;
    # success = the cube rests inside it.
    assert np.linalg.norm(final[:2] - zone[:2]) <= 0.06
    assert final[2] < 0.05  # actually resting near the table, not floating


def test_voice_sequence_pick_stack_and_place_at_corner():
    from rex_tendon.control.sim_primitives import SimArm, SimIntentExecutor
    from rex_tendon.control.voice_commands import handle_text_sequence

    arm = SimArm(viewer=False, arrange_objects=True, arrange_seed=2)
    try:
        executor = SimIntentExecutor(arm)
        colors = arm.scene_colors()

        out = handle_text_sequence(
            "take the red cube and put it on top of the purple", executor, colors
        )
        assert out[0].startswith("Picked up the red")
        assert "purple" in out[1]
        assert not executor.holding_object  # released after the stack attempt

        out = handle_text_sequence(
            "pick up the yellow cube and put it in the corner", executor, colors
        )
        assert out[0].startswith("Picked up the yellow")
        assert "corner" in out[1]
        # The cube must end near the corner point regardless of phrasing.
        corner = arm.resolve_location("corner")
        yellow = arm.find_object(color="yellow")
        import numpy as _np

        assert _np.linalg.norm(yellow.position(arm.data)[:2] - corner[:2]) <= 0.07
    finally:
        arm.close()


def _teleport(arm, obj, pos):
    jnt = arm.model.body_jntadr[obj.body_id]
    adr = arm.model.jnt_qposadr[jnt]
    arm.data.qpos[adr : adr + 3] = pos
    arm.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    vadr = arm.model.jnt_dofadr[jnt]
    if vadr >= 0:
        arm.data.qvel[vadr : vadr + 6] = 0.0
    import mujoco as mj

    mj.mj_forward(arm.model, arm.data)


def test_is_stacked_on_physical_truth(arm):
    red = arm.find_object(color="red")
    purple = arm.find_object(color="purple")
    rp = red.position(arm.data)
    # Exactly on top -> stacked.
    _teleport(arm, purple, rp + np.array([0, 0, red.half_height + purple.half_height]))
    assert arm._is_stacked_on(purple, red)
    # 5 cm away on the table -> NOT stacked (the old 6 cm place tolerance
    # would have called this a success).
    _teleport(arm, purple, rp + np.array([0.05, 0, 0]))
    assert not arm._is_stacked_on(purple, red)
    # On top but offset beyond the footprint -> NOT stacked.
    _teleport(
        arm,
        purple,
        rp + np.array([0.025, 0, red.half_height + purple.half_height]),
    )
    assert not arm._is_stacked_on(purple, red)


def test_staged_scene_is_clean_and_separated():
    from rex_tendon.control.sim_primitives import SimArm

    arm = SimArm(viewer=False, arrange_objects=True, arrange_seed=2)
    try:
        staged = [o for o in arm.objects if arm._in_workspace(o)]
        assert len(staged) == 4
        assert all(o.shape == "cube" for o in staged)
        import itertools

        for a, b in itertools.combinations(staged, 2):
            gap = np.linalg.norm(
                a.position(arm.data)[:2] - b.position(arm.data)[:2]
            )
            assert gap > 0.06  # 90-degree spacing on the ring
        for o in staged:
            # Clear of the 5 cm base pedestal (no startup explosion).
            assert np.linalg.norm(o.position(arm.data)[:2]) > 0.057
        assert arm.scene_colors() == {o.color for o in staged}
    finally:
        arm.close()


def test_stack_command_never_lies():
    """Whatever the physical outcome, the reported result must match the
    physical on-top truth check — the demo may fail, but it may not lie."""
    from rex_tendon.control.sim_primitives import SimArm

    arm = SimArm(viewer=False, arrange_objects=True, arrange_seed=3)
    try:
        colors = sorted(arm.scene_colors())
        top_color, base_color = colors[0], colors[1]
        ok = arm.stack(top_color, base_color)
        top = arm.find_object(color=top_color)
        base = arm.find_object(color=base_color)
        assert ok == arm._is_stacked_on(top, base)
    finally:
        arm.close()


def test_voice_intent_executor_pick(arm):
    from rex_tendon.control.sim_primitives import SimIntentExecutor
    from rex_tendon.control.voice_commands import handle_text

    executor = SimIntentExecutor(arm)
    out = handle_text(
        "pick up the red cube",
        executor,
        available_colors=arm.scene_colors(),
        holding_object=executor.holding_object,
    )
    assert out == "Picked up the red cube."
    assert executor.holding_object

    out = handle_text(
        "let go",
        executor,
        available_colors=arm.scene_colors(),
        holding_object=executor.holding_object,
    )
    assert out == "Released."
    assert not executor.holding_object

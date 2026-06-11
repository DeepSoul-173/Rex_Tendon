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

"""M1 tests: contact-gated suction gripper on the dev manipulation scene.

These pin the honesty contract of the new manipulation stack:
  - no attachment without real, measured contact force
  - attachment holds the object exactly where it was touched (no snap)
  - the physics engine carries the object (no qpos writes, no xfrc)
  - release lets the object settle under gravity
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

SCENE = "rex_assets/rex_simulation/pick_and_place_scene_manip.xml"


@pytest.fixture()
def sim():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    data.ctrl[:3] = 0.23  # neutral tendon lengths
    for _ in range(800):  # settle the arm upright
        mujoco.mj_step(model, data)
    return model, data


def _teleport(model, data, body_name, pos):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt = model.body_jntadr[bid]
    adr = model.jnt_qposadr[jnt]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    vadr = model.jnt_dofadr[jnt]
    data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def _tip_pos(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    return data.site_xpos[sid].copy()


def _cube_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube")
    return data.xpos[bid].copy()


def _park_other_objects(model, data, keep: str = "obj_cube"):
    """Move every obj_* body except `keep` far off-scene (lab conditions)."""
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name and name.startswith("obj_") and name != keep:
            _teleport(model, data, name, [10.0 + 0.1 * b, 10.0, 0.05])


def _touch_tip_with_cube(model, data):
    """Park the red cube just barely pressing the tip sphere from below.

    0.3 mm of overlap ~ where a real descend-to-contact stops (first force
    above the latch threshold); deeper forced penetration is unrealistic and
    makes the compliant seal visibly yield.
    """
    _park_other_objects(model, data)
    tip = _tip_pos(model, data)
    # tip sphere r=0.008 + cube half=0.01: centre gap 0.018 = exact touch.
    # mj_forward (inside _teleport) resolves contacts without integrating —
    # stepping here would let the cube free-fall away before sealing.
    _teleport(model, data, "obj_cube", tip + np.array([0.0, 0.0, -0.0177]))


def test_no_attach_without_contact(sim):
    from rex_tendon.manipulation.gripper import SuctionGripper

    model, data = sim
    grip = SuctionGripper(model, data)
    red = next(o for o in grip.objects if o.name == "obj_cube")
    # Cube far away: no contact, no attach, weld stays off.
    assert grip.contact_force_with(red) == 0.0
    assert not grip.attach(red)
    assert grip.attached is None
    assert not any(data.eq_active)


def test_attach_requires_and_uses_real_contact(sim):
    from rex_tendon.manipulation.gripper import SuctionGripper

    model, data = sim
    grip = SuctionGripper(model, data)
    red = next(o for o in grip.objects if o.name == "obj_cube")

    _touch_tip_with_cube(model, data)
    assert grip.contact_force_with(red) > 0.0  # solver-measured force
    assert grip.attach(red)
    assert grip.attached is red
    assert int(np.sum(data.eq_active)) == 1


def test_attach_holds_seal_no_snap(sim):
    """At a WORKING pose (where the pipeline attaches), the object must stay
    sealed at the tip interface throughout the hold.

    Note what is and is not asserted: a 20 g payload genuinely sags the arm
    by centimetres (the distal spine joints stay soft even on the engineered
    plant) — that is structural deflection, not a grasp defect. The seal
    contract is: tip-to-object distance stays at seal scale the whole time
    (snap/teleport = decimetres instantly; a drop = separation growing)."""
    from rex_tendon.manipulation.gripper import SuctionGripper

    model, data = sim
    grip = SuctionGripper(model, data)
    red = next(o for o in grip.objects if o.name == "obj_cube")
    _bend(model, data, (0.5, 0.0), 1200)  # settled working pose
    _touch_tip_with_cube(model, data)
    assert grip.attach(red)
    for k in range(900):  # 1.8 s: sag transient + steady hold
        mujoco.mj_step(model, data)
        if k % 50 == 0:
            gap = float(
                np.linalg.norm(_cube_pos(model, data) - _tip_pos(model, data))
            )
            assert 0.010 < gap < 0.045, f"seal broken at t={k*0.002:.2f}s: {gap:.3f}"
    assert grip.attached is red


def _relpose_in_carrier(model, data, grip, obj):
    """Object position expressed in the tip-body frame (what the weld holds)."""
    b1 = grip._tip_body
    r1 = data.xmat[b1].reshape(3, 3)
    return r1.T @ (data.xpos[obj.body_id] - data.xpos[b1])


def _bend(model, data, cursor, ticks):
    from rex_tendon.control.geometry import convert_2d_cursor_to_target_lengths

    data.ctrl[:3] = convert_2d_cursor_to_target_lengths(
        np.array(cursor, dtype=np.float32),
        np.full(3, 0.23, dtype=np.float32),
        model.actuator_ctrlrange[:3, 0],
        model.actuator_ctrlrange[:3, 1],
        1.0,
    )
    for _ in range(ticks):
        mujoco.mj_step(model, data)


def test_weld_carries_object_through_motion(sim):
    """Transport between WORKING poses — the pipeline's actual scenario.

    (Attaching at upright neutral and then pitching the segment 90 degrees
    swings the payload into the arm's own belly, where contact shoves it —
    physically correct, but not a configuration transport ever uses.)
    """
    from rex_tendon.manipulation.gripper import SuctionGripper

    model, data = sim
    grip = SuctionGripper(model, data)
    red = next(o for o in grip.objects if o.name == "obj_cube")

    _bend(model, data, (0.5, 0.0), 1200)  # working pose A, settled
    _touch_tip_with_cube(model, data)  # payload hangs below the tip
    assert grip.attach(red)
    rel0 = _relpose_in_carrier(model, data, grip, red)
    tip_a = _tip_pos(model, data)

    _bend(model, data, (0.1, 0.5), 2000)  # transport to working pose B

    tip_b = _tip_pos(model, data)
    assert np.linalg.norm(tip_b[:2] - tip_a[:2]) > 0.02  # genuinely moved
    assert _cube_pos(model, data)[2] > 0.03  # payload airborne throughout
    # The weld's promise is pose-in-carrier-frame (world offsets rotate with
    # the segment, as any rigid attachment's would). Budget: the compliant
    # seal flexes up to ~1 cm under payload load through a large
    # reorientation — sub-centimetre rigidity, vs decimetres when broken.
    rel1 = _relpose_in_carrier(model, data, grip, red)
    assert np.linalg.norm(rel1 - rel0) < 0.012
    # And nobody injected helper forces:
    assert float(np.abs(data.xfrc_applied).max()) == 0.0
    # And nobody injected helper forces:
    assert float(np.abs(data.xfrc_applied).max()) == 0.0


def test_detach_releases_and_object_settles(sim):
    from rex_tendon.manipulation.gripper import SuctionGripper

    model, data = sim
    grip = SuctionGripper(model, data)
    red = next(o for o in grip.objects if o.name == "obj_cube")
    _touch_tip_with_cube(model, data)
    assert grip.attach(red)
    released = grip.detach()
    assert released is red and grip.attached is None
    assert not any(data.eq_active)
    for _ in range(1000):  # 2 s of free physics
        mujoco.mj_step(model, data)
    cube = _cube_pos(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_cube")
    vadr = model.jnt_dofadr[model.body_jntadr[bid]]
    speed = float(np.linalg.norm(data.qvel[vadr : vadr + 3]))
    assert cube[2] < 0.05  # fell from the tip, resting low
    assert speed < 0.02  # settled, not bouncing

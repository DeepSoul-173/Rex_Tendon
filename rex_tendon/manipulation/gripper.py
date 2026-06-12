"""Contact-gated suction gripper — the honest replacement for magnetic pickup.

Model: the tip is a suction/adhesion end-effector. Attachment requires REAL,
measured contact force between the tip sphere and the object (MuJoCo contact
solver, not proximity), and is implemented by activating the scene's
pre-declared weld equality constraint for that object. The physics engine
then carries the object — no teleporting, no zeroed velocities, no hidden
anti-gravity forces, anywhere. Release deactivates the weld and the object
settles under gravity like a real dropped part.

The welds (grasp_* equalities, inactive by default) have shipped with the
scene all along; the legacy controllers bypassed them with kinematic hacks.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from ..perception.scene_objects import SceneObject, discover_objects

# Real normal force (N) the tip must exert on the object before suction can
# latch. Low enough for a gentle touch, high enough that grazing doesn't count.
GRASP_FORCE_THRESHOLD = 0.02

# Suction interface stiffness (MuJoCo solref: timeconst, dampratio). The
# default (0.02, 1) acts like a ~50 N/m tether for a 20 g payload — measured:
# the carried cube swings decimetres and back-reacts the arm into sustained
# oscillation. With the relpose captured at the touch pose (near-zero initial
# violation) a much stiffer seal is stable and carries rigidly.
SUCTION_SOLREF = (0.004, 1.0)

# Seal standoff (m): the held pose backs the object off the tip sphere by a
# lip's thickness. Contact is REQUIRED to seal, but holding the bodies in
# permanent interpenetration makes the contact and weld constraints fight
# (measured: vibration + drift). 3 mm keeps the pair contact-free even while
# the compliant seal flexes under payload sag-bounce (measured: 1.5 mm
# re-contacted during the post-attach transient and ratcheted the pose).
SEAL_STANDOFF = 0.003


class SuctionGripper:
    """Attach/detach objects via contact-verified weld constraints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tip_geom: str = "tip_contact",
    ):
        self.model = model
        self.data = data
        self.tip_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, tip_geom
        )
        if self.tip_geom_id < 0:
            raise ValueError(f"tip geom '{tip_geom}' not found in model")
        tip_body = int(model.geom_bodyid[self.tip_geom_id])
        self._tip_body = tip_body

        self.objects = discover_objects(model)
        # Map object body -> weld equality id, discovered structurally (the
        # constraint names don't always match the body names).
        self._eq_for_body: dict[int, int] = {}
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_WELD:
                continue
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            if b1 == tip_body:
                self._eq_for_body[b2] = e
            elif b2 == tip_body:
                self._eq_for_body[b1] = e

        # Compliant suction interface on every grasp weld (see SUCTION_SOLREF).
        for e in self._eq_for_body.values():
            self.model.eq_solref[e] = SUCTION_SOLREF

        self.attached: Optional[SceneObject] = None

    # ── Contact sensing ─────────────────────────────────────────────────────────

    def contact_force_with(self, obj: SceneObject) -> float:
        """Total normal force (N) currently exchanged between tip and object."""
        total = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {int(con.geom1), int(con.geom2)}
            if pair == {self.tip_geom_id, obj.geom_id}:
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                total += abs(float(wrench[0]))  # normal component
        return total

    def in_contact(self, obj: SceneObject) -> bool:
        return self.contact_force_with(obj) >= GRASP_FORCE_THRESHOLD

    # ── Attach / detach ─────────────────────────────────────────────────────────

    def can_grasp(self, obj: SceneObject) -> bool:
        return obj.body_id in self._eq_for_body and self.in_contact(obj)

    def _write_current_relpose(self, eq: int, obj: SceneObject) -> None:
        """Point the weld at the CURRENT tip->object relative pose.

        MuJoCo resolves a zero relpose at COMPILE time (qpos0), not at
        activation — activating the stock weld yanks the object toward its
        XML spawn pose relative to the tip and explodes the sim (measured).
        Writing eq_data here makes the weld hold the object exactly where it
        was touched.
        """
        b1, b2 = self._tip_body, obj.body_id
        # body2 position in body1 frame, backed off by the seal standoff so
        # the held pose is contact-free (see SEAL_STANDOFF).
        tip_center = self.data.geom_xpos[self.tip_geom_id]
        away = self.data.xpos[b2] - tip_center
        away_norm = float(np.linalg.norm(away))
        standoff_world = (
            away / away_norm * SEAL_STANDOFF if away_norm > 1e-9 else np.zeros(3)
        )
        r1 = self.data.xmat[b1].reshape(3, 3)
        relpos = r1.T @ (self.data.xpos[b2] + standoff_world - self.data.xpos[b1])
        # body2 orientation in body1 frame: q1^-1 * q2
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, self.data.xquat[b1])
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, q1_inv, self.data.xquat[b2])

        self.model.eq_data[eq][:] = 0.0
        self.model.eq_data[eq][3:6] = relpos
        self.model.eq_data[eq][6:10] = relquat
        self.model.eq_data[eq][10] = 1.0  # torquescale

    def attach(self, obj: SceneObject) -> bool:
        """Latch suction onto `obj` — only with real contact force present.

        The weld is re-anchored to the relative pose at this instant, so the
        object is held exactly where it was touched: no snap, no jump.
        """
        if self.attached is not None:
            return False
        eq = self._eq_for_body.get(obj.body_id)
        if eq is None or not self.in_contact(obj):
            return False
        self._write_current_relpose(eq, obj)
        self.data.eq_active[eq] = 1
        self.attached = obj
        return True

    def detach(self) -> Optional[SceneObject]:
        """Release suction; the object is now free and settles by physics."""
        if self.attached is None:
            return None
        eq = self._eq_for_body[self.attached.body_id]
        self.data.eq_active[eq] = 0
        released, self.attached = self.attached, None
        return released

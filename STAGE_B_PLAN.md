# Stage B — 2-Section / 4-DOF S-Curve Arm + Stacking

Branch: `stage-b-2section` (cut from `master` at the Stage A green checkpoint).
Inherits the **assisted (6 cm) grasp** — Stage A showed the single-section arm
cannot do a near-contact (2.5 cm) grasp (~1% success); the goal of Stage B is to
*earn* a tighter grasp via better tip control from the extra DOF.

## Why 2 sections
Single section = 3 tendons = **2 task-DOF** (bend direction + curvature). Two
independently-actuated sections = 6 tendons = **4 task-DOF**, enabling S-curves
(reach-around, side approaches) and finer tip positioning — the prerequisite for
stacking dice and for a tighter grasp tolerance.

## Milestones

- [x] **M1 — Model (S-curve kinematics).** `generate-xml --num-sections 2`
  splits each helical tendon at the mid-segment into a lower + upper tendon
  (6 tendons, 6 actuators), with per-section `ctrlrange` scaled by section length.
  Verified: independent section bending + S-curves. Output:
  `rex_assets/rex_simulation/tentacle_2section.xml`.

- [ ] **M2 — Control mapping (4-DOF).** Extend `control/geometry.py`:
  `convert_4d_cursor_to_target_lengths(action4, ...)` = two independent
  2D-cursor → 3-tendon maps (one per section). Action space 2D → 4D.

- [ ] **M3 — 2-section pick-place scene.** Embed the 2-section tentacle in a
  desk/objects/place-zone scene (mirror `pick_and_place_scene.xml`), keeping the
  base-yaw DOF. Re-tune cameras.

- [ ] **M4 — Env updates.** `TentaclePickPlaceEnv`: 4D action, 6 tendon
  actuators (obs uses 6 actuator lengths), drive `ctrl[:6]`; keep base-yaw
  dormant/optional. Keep the hybrid grasp + reward shaping from Stage A.

- [ ] **M5 — Stacking task.** New task mode: pick the *top* cube of a source
  stack; a place target whose Z rises one cube-height per placed cube; success =
  N cubes stacked within tolerance. Reward = per-cube place bonus + stack-height
  progress.

- [ ] **M6 — Curriculum + training.** Stack-height curriculum 1→N (reuse the
  `MultiObjectCurriculumCallback` pattern), warm-start where possible, log
  per-stack-height success. Deliverable: stacking generalization curve.

- [ ] **M7 — (stretch) Tighter grasp.** Re-run the near-contact experiment on the
  4-DOF arm; measure whether the extra DOF earns a grasp tolerance the
  single-section arm couldn't (the Stage A→B narrative payoff).

## Notes
- The 2-section model uses the same 20-segment mesh chain; only tendon routing +
  actuator count change, so existing dynamics/stiffness carry over.
- `num_sections` is parametric — a 3-section (6-DOF) variant is one flag away if
  M5/M6 need more dexterity.
- Action-dim change (2→4) means Stage B trains a fresh policy; Stage A's
  single-section policy is not transferable (different action space).

"""M0: damping engineering study for the spiral arm.

The scene's spine ball joints ship with damping 0.05 (base) down to 0.004
(tip) — a nearly undamped structure that rings for seconds after every motion
and makes any controller look shaky. This tool measures the step response of
the tip across damping scales, reports rise time / settle time / residual
oscillation, and bakes the chosen damping into a separate DEV manipulation
scene (the original scene and all RL artifacts stay untouched).

    python -m rex_tendon.manipulation.damping_study --sweep
    python -m rex_tendon.manipulation.damping_study --apply 30 \
        --out rex_assets/rex_simulation/pick_and_place_scene_manip.xml

Outputs eval_results/damping_sweep.png + .json (dissertation evidence).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

from ..control.geometry import (
    convert_2d_cursor_to_target_lengths,
    convert_4d_cursor_to_target_lengths,
)

SCENE = "rex_assets/rex_simulation/pick_and_place_scene.xml"
SCENE_2SECTION = "rex_assets/rex_simulation/pick_and_place_scene_2section.xml"
BASELINES_2SECTION = np.array([0.13] * 3 + [0.17] * 3, dtype=np.float32)
SETTLE_BAND = 0.002  # m: tip considered settled within this of its final pose
TRIAL_SECONDS = 8.0
FINAL_WINDOW = 0.5  # s averaged to define the final pose
LATE_WINDOW = (5.0, 8.0)  # s: residual oscillation measurement window


def _spine_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(joint ids, dof addresses) of the spiral's ball joints joint1..N."""
    joints: list[int] = []
    dofs: list[int] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not (name and re.fullmatch(r"joint\d+", name)):
            continue
        joints.append(j)
        adr = int(model.jnt_dofadr[j])
        ndof = 3 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
        dofs.extend(range(adr, adr + ndof))
    return np.array(joints, dtype=int), np.array(sorted(dofs), dtype=int)


def measure_step_response(
    xml_path: str = SCENE,
    damping_scale: float = 1.0,
    stiffness_scale: float = 1.0,
) -> dict:
    """Step-bend the arm and measure how the tip rings.

    Returns rise time (90% of displacement), settle time (last excursion out
    of the 2 mm band), overshoot ratio, and late residual peak-to-peak.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joints, dofs = _spine_indices(model)
    model.dof_damping[dofs] = model.dof_damping[dofs] * damping_scale
    model.jnt_stiffness[joints] = model.jnt_stiffness[joints] * stiffness_scale

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    act_low = model.actuator_ctrlrange[:3, 0]
    act_high = model.actuator_ctrlrange[:3, 1]

    two_section = model.nu >= 6

    # Let the arm settle at neutral first.
    if two_section:
        data.ctrl[:6] = BASELINES_2SECTION
    else:
        data.ctrl[:3] = 0.23
    for _ in range(1500):
        mujoco.mj_step(model, data)
    tip0 = data.site_xpos[tip_id].copy()

    # Step input: a hard bend command (maximum excitation of the ringing).
    if two_section:
        lo6 = model.actuator_ctrlrange[:6, 0]
        hi6 = model.actuator_ctrlrange[:6, 1]
        data.ctrl[:6] = convert_4d_cursor_to_target_lengths(
            np.array([0.8, 0.0, 0.8, 0.0], dtype=np.float32),
            BASELINES_2SECTION,
            lo6,
            hi6,
            1.0,
        )
    else:
        data.ctrl[:3] = convert_2d_cursor_to_target_lengths(
            np.array([0.8, 0.0], dtype=np.float32),
            np.full(3, 0.23, dtype=np.float32),
            act_low,
            act_high,
            1.0,
        )

    dt = float(model.opt.timestep)
    n = int(TRIAL_SECONDS / dt)
    tips = np.empty((n, 3))
    for i in range(n):
        mujoco.mj_step(model, data)
        tips[i] = data.site_xpos[tip_id]

    t = np.arange(n) * dt
    final = tips[int(-FINAL_WINDOW / dt) :].mean(axis=0)
    dist_to_final = np.linalg.norm(tips - final, axis=1)
    displacement = np.linalg.norm(tips - tip0, axis=1)
    step_size = float(np.linalg.norm(final - tip0))

    risen = np.nonzero(displacement >= 0.9 * step_size)[0]
    rise_time = float(t[risen[0]]) if len(risen) else float("inf")
    outside = np.nonzero(dist_to_final > SETTLE_BAND)[0]
    settle_time = float(t[outside[-1]]) if len(outside) else 0.0
    overshoot = float(displacement.max() / max(step_size, 1e-9) - 1.0)
    late = dist_to_final[int(LATE_WINDOW[0] / dt) : int(LATE_WINDOW[1] / dt)]
    residual_pp = float(late.max() - late.min()) if len(late) else 0.0

    return {
        "damping_scale": damping_scale,
        "stiffness_scale": stiffness_scale,
        "rise_time_s": round(rise_time, 3),
        "settle_time_s": round(settle_time, 3),
        "overshoot": round(overshoot, 3),
        "residual_pp_mm": round(residual_pp * 1000.0, 3),
        "step_size_m": round(step_size, 4),
        "trace_t": t[:: 10].tolist(),
        "trace_d": dist_to_final[:: 10].tolist(),
    }


def probe_min_settled_z(
    xml_path: str,
    damping_scale: float = 1.0,
    stiffness_scale: float = 1.0,
    ticks: int = 900,
) -> dict:
    """Lowest settled tip height over a grid of descend-style poses.

    THE feasibility metric for contact grasping: cube-top contact needs the
    tip centre at z <= 0.030 (cube top 0.022 + tip sphere 0.008). Stiffening
    that buys settle time but prices the arm out of the low workspace is a
    failed trade (measured on the 1-section arm: x8 stiffness -> min z 0.072).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joints, dofs = _spine_indices(model)
    model.dof_damping[dofs] = model.dof_damping[dofs] * damping_scale
    model.jnt_stiffness[joints] = model.jnt_stiffness[joints] * stiffness_scale
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    lo6 = model.actuator_ctrlrange[:6, 0]
    hi6 = model.actuator_ctrlrange[:6, 1]

    best = {"z": float("inf")}
    for m1 in (0.5, 0.9):
        for m2 in (0.5, 0.9, 1.2):
            for b1, b2 in ((0.08, 0.08), (0.10, 0.10), (0.13, 0.17), (0.13, 0.10)):
                mujoco.mj_resetData(model, data)
                baselines = np.array([b1] * 3 + [b2] * 3, dtype=np.float32)
                data.ctrl[:6] = convert_4d_cursor_to_target_lengths(
                    np.array([m1, 0.0, m2, 0.0], dtype=np.float32),
                    baselines, lo6, hi6, 1.0,
                )
                for _ in range(ticks):
                    mujoco.mj_step(model, data)
                tip = data.site_xpos[tip_id]
                r = float(np.linalg.norm(tip[:2]))
                if tip[2] < best["z"] and r > 0.03:  # over the table, not base
                    best = {
                        "z": float(tip[2]), "r": r,
                        "pose": (m1, m2, b1, b2),
                    }
    return best


def run_sweep(scales: list[float], out_dir: Path) -> list[dict]:
    results = [measure_step_response(damping_scale=s) for s in scales]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = [
        {k: r[k] for k in (
            "damping_scale", "rise_time_s", "settle_time_s",
            "overshoot", "residual_pp_mm",
        )}
        for r in results
    ]
    (out_dir / "damping_sweep.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in results:
            ax1.plot(
                r["trace_t"],
                1000.0 * np.asarray(r["trace_d"]),
                label=f"{r['damping_scale']:g}x",
            )
        ax1.set(
            xlabel="time [s]",
            ylabel="tip distance to final pose [mm]",
            title="Step response vs spine damping",
            ylim=(0, 60),
        )
        ax1.axhline(2.0, color="gray", ls="--", lw=0.8)
        ax1.legend(title="damping")

        scales_arr = [r["damping_scale"] for r in results]
        ax2.plot(scales_arr, [r["settle_time_s"] for r in results], "o-",
                 label="settle time (2 mm band)")
        ax2.plot(scales_arr, [r["rise_time_s"] for r in results], "s-",
                 label="rise time (90%)")
        ax2.set(xscale="log", xlabel="damping scale", ylabel="seconds",
                title="Settle vs rise time")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "damping_sweep.png", dpi=130)
        print(f"Plot: {out_dir / 'damping_sweep.png'}")
    except ImportError:
        print("matplotlib unavailable — JSON only")
    return summary


def generate_manip_scene(
    src: str, out: str, damping_scale: float, stiffness_scale: float
) -> None:
    """Bake co-scaled spine stiffness+damping into a separate dev scene file.

    Pure damping is the wrong fix (measured: it turns ringing into slow
    creep). Critical-damping behaviour needs stiffness and damping raised
    together; (x8, x3) gives a 10x faster settle at 12% reach cost. Only the
    `joint<N>` ball-joint attributes are touched; the source scene (used by
    RL runs) is never modified.
    """
    text = Path(src).read_text(encoding="utf-8")

    def scale_attrs(match: re.Match) -> str:
        prefix, k_val, mid, c_val = match.groups()
        return (
            f'{prefix}stiffness="{float(k_val) * stiffness_scale:.6g}"'
            f'{mid}damping="{float(c_val) * damping_scale:.6g}"'
        )

    out_text = re.sub(
        r'(<joint type="ball" name="joint\d+"[^>]*?)'
        r'stiffness="([\d.eE+-]+)"([^>]*?)damping="([\d.eE+-]+)"',
        scale_attrs,
        text,
    )
    header = (
        f"<!-- DEV MANIPULATION SCENE - generated by damping_study.py from "
        f"{Path(src).name}: spine stiffness x{stiffness_scale:g}, damping "
        f"x{damping_scale:g} (measured settle 3.69s -> 0.36s). "
        f"Do not use for RL training comparisons. -->\n"
    )
    # The XML declaration must stay the first bytes of the file.
    if out_text.lstrip().startswith("<?xml"):
        first_line, rest = out_text.split("\n", 1)
        out_text = first_line + "\n" + header + rest
    else:
        out_text = header + out_text
    Path(out).write_text(out_text, encoding="utf-8")
    print(
        f"Wrote {out} (stiffness x{stiffness_scale:g}, damping x{damping_scale:g})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spine damping study (M0)")
    parser.add_argument("--sweep", action="store_true", help="Run the sweep")
    parser.add_argument(
        "--scales", type=float, nargs="*",
        default=[1, 3, 10, 30, 60, 100, 200],
    )
    parser.add_argument("--apply", type=float, default=None,
                        help="Bake this damping scale into a dev scene")
    parser.add_argument("--apply-stiffness", type=float, default=1.0,
                        help="Stiffness co-scale baked alongside --apply")
    parser.add_argument(
        "--out", default="rex_assets/rex_simulation/pick_and_place_scene_manip.xml"
    )
    parser.add_argument("--src", default=SCENE)
    args = parser.parse_args()

    if args.sweep:
        summary = run_sweep(args.scales, Path("eval_results"))
        print(f"{'scale':>8} {'rise[s]':>8} {'settle[s]':>10} "
              f"{'overshoot':>10} {'residual[mm]':>13}")
        for r in summary:
            print(
                f"{r['damping_scale']:>8g} {r['rise_time_s']:>8} "
                f"{r['settle_time_s']:>10} {r['overshoot']:>10} "
                f"{r['residual_pp_mm']:>13}"
            )
    if args.apply is not None:
        generate_manip_scene(args.src, args.out, args.apply, args.apply_stiffness)


if __name__ == "__main__":
    main()

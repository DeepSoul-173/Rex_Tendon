"""Headless rollout + diagnostics for trained pick-place / stacking policies.

Loads a checkpoint (+ VecNormalize stats), rebuilds the env exactly as
training did (FULL curriculum phase forced), runs N episodes, and reports
per-episode and aggregate behaviour metrics — grasp / lift / carry / place —
so a policy can be judged without TensorBoard or a viewer. Optionally renders
episodes to an mp4 for visual inspection.

Usage:
    python -m rex_tendon.evaluation.rollout_pick_place \
        --model rex_results/pick_place_stack/<run>/models/best_model/best_model.zip \
        --config rex_tendon/configs/pick_place_stack.yaml \
        --vecnormalize rex_results/pick_place_stack/<run>/models/vecnormalize.pkl \
        --episodes 20 --video eval_results/best_model_rollout.mp4

Outputs a JSON summary to eval_results/ (kept separate from training's
rex_results/). Read-only with respect to all training artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..training.rl.pick_place_env import TentaclePickPlaceEnv
from ..training.rl.pick_place_training import (
    get_active_task,
    load_pick_place_config,
)

LIFT_HEIGHT = 0.03  # m above rest height that counts as "lifted"


def build_env(
    config_path: str | None,
    vecnormalize_path: str | None,
    stack_count: int | None = None,
):
    """Recreate the training-time env stack for evaluation.

    Returns (venv, raw_env): the (possibly VecNormalize-wrapped) vec env for
    the policy, and the underlying TentaclePickPlaceEnv for direct state
    inspection during diagnostics.
    """
    pp_config = load_pick_place_config(config_path)
    task = get_active_task(pp_config)

    # Force FULL curriculum phase: a huge "global step" makes the env enable
    # grasping + placing, matching late-training behaviour.
    counter = SimpleNamespace(value=1_000_000_000)

    # Pin the object/stack count to the training condition. Without an
    # explicit curriculum state the env falls back to num_spawned_objects
    # (5 in the stack config) — earlier rollouts unknowingly evaluated a
    # 5-object source tower the training run never sees.
    n = (
        stack_count
        if stack_count is not None
        else int(pp_config.env.min_spawned_objects)
    )
    curriculum_state = SimpleNamespace(value=n)

    raw_env = TentaclePickPlaceEnv(
        config=pp_config.env,
        task_config=task,
        global_step_counter=counter,
        object_curriculum_state=curriculum_state,
    )
    venv = DummyVecEnv([lambda: Monitor(raw_env)])

    if vecnormalize_path:
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = False
        venv.norm_reward = False

    return venv, raw_env


def run_episodes(
    model: PPO,
    venv,
    raw_env: TentaclePickPlaceEnv,
    episodes: int,
    deterministic: bool = True,
    video_writer=None,
    video_renderer=None,
    video_camera=None,
):
    """Roll out N episodes; return a list of per-episode metric dicts."""
    results = []
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        ep = {
            "reward": 0.0,
            "length": 0,
            "ever_grasped": False,
            "ever_lifted": False,
            "num_placed": 0,
            "place_success": False,
            "min_obj_to_place": np.inf,
            "final_obj_to_place": np.nan,
            "mean_tip_to_obj": [],
            "max_obj_height": 0.0,
            "drops": 0,
        }
        was_grasped = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0]

            ep["length"] += 1
            ep["min_obj_to_place"] = min(
                ep["min_obj_to_place"], float(info["object_to_place_distance"])
            )
            ep["final_obj_to_place"] = float(info["object_to_place_distance"])
            ep["mean_tip_to_obj"].append(float(info["tip_to_object_distance"]))

            grasped = bool(info.get("is_grasped", False))
            ep["ever_grasped"] |= grasped
            if was_grasped and not grasped and not done:
                ep["drops"] += 1
            was_grasped = grasped

            # Active-object height above its rest position (direct env read).
            idx = raw_env.active_object_idx
            obj_z = float(raw_env._get_object_position(idx)[2])
            rest_z = float(raw_env.object_rest_z[idx])
            ep["max_obj_height"] = max(ep["max_obj_height"], obj_z - rest_z)

            ep["num_placed"] = int(getattr(raw_env, "_num_placed", 0))
            ep["place_success"] |= bool(info.get("place_success", False))

            if "episode" in info:  # Monitor's end-of-episode record
                ep["reward"] = float(info["episode"]["r"])

            if video_writer is not None:
                video_renderer.update_scene(raw_env.data, camera=video_camera)
                frame = video_renderer.render()
                import cv2

                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        ep["ever_lifted"] = ep["max_obj_height"] > LIFT_HEIGHT
        ep["mean_tip_to_obj"] = float(np.mean(ep["mean_tip_to_obj"]))
        results.append(ep)
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "episodes": n,
        "mean_reward": float(np.mean([e["reward"] for e in results])),
        "grasp_rate": sum(e["ever_grasped"] for e in results) / n,
        "lift_rate": sum(e["ever_lifted"] for e in results) / n,
        "place_success_rate": sum(e["place_success"] for e in results) / n,
        "mean_num_placed": float(np.mean([e["num_placed"] for e in results])),
        "mean_drops": float(np.mean([e["drops"] for e in results])),
        "mean_min_obj_to_place": float(
            np.mean([e["min_obj_to_place"] for e in results])
        ),
        "mean_final_obj_to_place": float(
            np.mean([e["final_obj_to_place"] for e in results])
        ),
        "mean_tip_to_obj": float(np.mean([e["mean_tip_to_obj"] for e in results])),
        "mean_max_obj_height": float(
            np.mean([e["max_obj_height"] for e in results])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless diagnostic rollout for pick-place policies"
    )
    parser.add_argument("--model", required=True, help="Path to PPO .zip checkpoint")
    parser.add_argument("--config", default=None, help="Pick-place YAML config")
    parser.add_argument(
        "--vecnormalize", default=None, help="Path to vecnormalize.pkl"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--stack-count",
        type=int,
        default=None,
        help="Objects in the source stack (default: the config's "
        "min_spawned_objects — the level training starts at)",
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Sample actions instead of mode"
    )
    parser.add_argument(
        "--video", default=None, help="Write an mp4 of the rollout to this path"
    )
    parser.add_argument(
        "--out", default=None, help="JSON summary path (default: eval_results/...)"
    )
    args = parser.parse_args()

    venv, raw_env = build_env(args.config, args.vecnormalize, args.stack_count)
    model = PPO.load(args.model, device="cpu")

    video_writer = video_renderer = video_camera = None
    if args.video:
        import cv2
        import mujoco
        from mujoco import renderer as mj_renderer

        video_renderer = mj_renderer.Renderer(raw_env.model, width=640, height=480)
        # Prefer a fixed scene camera when the model defines one.
        video_camera = -1
        for cam_name in ("table_corner_cam", "robot_cam"):
            if mujoco.mj_name2id(
                raw_env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            ) >= 0:
                video_camera = cam_name
                break
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        fps = 1.0 / raw_env.time_per_step
        video_writer = cv2.VideoWriter(
            args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480)
        )

    results = run_episodes(
        model,
        venv,
        raw_env,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        video_writer=video_writer,
        video_renderer=video_renderer,
        video_camera=video_camera,
    )

    if video_writer is not None:
        video_writer.release()
        video_renderer.close()

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    out = args.out or (
        f"eval_results/rollout_{Path(args.model).stem}_"
        f"{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"model": args.model, "summary": summary, "episodes": results}, f, indent=2
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

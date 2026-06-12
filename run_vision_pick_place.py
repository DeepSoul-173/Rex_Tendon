"""Vision-based pick-and-place demo with YOLO and a trained PPO policy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Please install ultralytics: pip install ultralytics") from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as exc:
    raise SystemExit(
        "Please install stable_baselines3: pip install stable-baselines3"
    ) from exc

import mujoco

from rex_tendon.configs.pick_place_config import PickPlaceConfig
from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv
from rex_tendon.training.rl.pick_place_training import load_pick_place_config

Box2D = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-based Pick and Place with YOLO")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained RL model (defaults to the most recent run under rex_results/pick_place)",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default=None,
        help="YOLO weights. Default: newest fine-tuned yolo_training/*/weights/"
        "best.pt; falls back to COCO yolov8n.pt (which does NOT know the cubes).",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    return parser.parse_args()


def resolve_model_path(explicit: Optional[str]) -> Optional[Path]:
    """Return an explicit model path, or auto-discover the most recent run."""
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    base = Path("rex_results/pick_place")
    if not base.exists():
        return None
    model_dirs = [d for d in base.glob("*/models") if d.is_dir()]
    if not model_dirs:
        return None
    latest = max(model_dirs, key=lambda d: d.stat().st_mtime)
    for candidate in (latest / "best_model" / "best_model.zip", latest / "final_model.zip"):
        if candidate.exists():
            return candidate
    zips = sorted(latest.glob("*.zip"))
    return zips[-1] if zips else None


def resolve_vecnormalize(model_path: Path) -> Optional[Path]:
    """Find vecnormalize stats next to the model or in the run's models/ dir."""
    for directory in (model_path.parent, model_path.parent.parent):
        candidate = directory / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    return None


def reset_active_env(active_env: Any) -> np.ndarray:
    """Normalize Gymnasium and VecEnv reset return shapes."""
    reset_result = active_env.reset()
    if isinstance(reset_result, tuple):
        return np.asarray(reset_result[0], dtype=np.float32)
    return np.asarray(reset_result, dtype=np.float32)


def step_active_env(active_env: Any, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Normalize Gymnasium and VecEnv step return shapes."""
    step_result = active_env.step(action)
    if len(step_result) == 4:
        obs, _reward, dones, infos = step_result
        done = bool(np.asarray(dones).reshape(-1)[0])
        info = dict(infos[0]) if infos else {}
        return np.asarray(obs, dtype=np.float32), done, info

    obs, _reward, terminated, truncated, info = step_result
    return np.asarray(obs, dtype=np.float32), bool(terminated or truncated), dict(info)


def first_detection(frame_bgr: np.ndarray, detector: Any) -> Optional[Box2D]:
    """Return the first YOLO bounding box as integer pixel coordinates."""
    results = detector.predict(frame_bgr, verbose=False, conf=0.25)
    if not results:
        return None

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = getattr(boxes[0], "xyxy", None)
    if xyxy is None:
        return None

    coords = np.asarray(xyxy[0], dtype=np.float32).reshape(4)
    return tuple(int(v) for v in coords)


def build_env(config_path: Optional[str]) -> TentaclePickPlaceEnv:
    if config_path:
        config = load_pick_place_config(config_path)
    else:
        config = PickPlaceConfig()

    config.env.curriculum_enabled = False
    env = TentaclePickPlaceEnv(config=config.env, render_mode="rgb_array")
    env.image_size = (480, 480)
    if env.renderer is not None:
        env.renderer.close()
    env.renderer = mujoco.Renderer(env.model, height=480, width=480)
    return env


def main() -> None:
    args = parse_args()
    rl_model_path = resolve_model_path(args.model_path)
    if rl_model_path is None:
        print(
            "Error: no trained RL model found. Pass --model_path or train one with "
            "`python -m rex_tendon pick-place train` first."
        )
        return

    print(f"Loading RL model from {rl_model_path}...")
    rl_model = PPO.load(str(rl_model_path))

    print("Initializing MuJoCo environment...")
    env = build_env(args.config)
    assert env.renderer is not None

    active_env: Any = env
    stats_path = resolve_vecnormalize(rl_model_path)
    if stats_path is not None:
        print(f"Loading normalization stats from {stats_path}...")
        vec_env = DummyVecEnv([lambda: env])
        active_env = VecNormalize.load(str(stats_path), vec_env)
        active_env.training = False
        active_env.norm_reward = False
    else:
        print("Warning: no VecNormalize stats found; policy may behave poorly.")

    yolo_weights = args.yolo_model
    if yolo_weights is None:
        trained = sorted(
            Path("yolo_training").glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if trained:
            yolo_weights = str(trained[-1])
            print(f"Using fine-tuned detector: {yolo_weights}")
        else:
            yolo_weights = "yolov8n.pt"
            print(
                "WARNING: no fine-tuned detector found — using COCO yolov8n.pt, "
                "which does NOT know these cubes. Train one with:\n"
                "  python -m rex_tendon.training.vision.generate_synthetic_data "
                "--count 1200 --out datasets/cubes_synth --size 480\n"
                "  python -m rex_tendon vision train datasets/cubes_synth/"
                "dataset.yaml --config rex_tendon/configs/vision_synth.yaml"
            )
    print("Loading YOLO model...")
    yolo_detector = YOLO(yolo_weights)

    window_name = "YOLO Vision - Pick & Place"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)

    try:
        print("\nStarting simulation. Press 'q' to quit.")
        for episode in range(args.episodes):
            obs = reset_active_env(active_env)
            done = False
            base_y_bottom: Optional[int] = None
            grasp_buffer = 0
            ground_calibration_frames = 0

            print(f"\n--- Starting Episode {episode + 1} ---")
            while not done:
                step_started = time.perf_counter()
                action, _ = rl_model.predict(obs, deterministic=True)
                obs, done, info = step_active_env(active_env, np.asarray(action))

                env.renderer.update_scene(env.data, camera="workspace_cam")
                frame = env.render()
                if frame is None:
                    continue

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.resize(frame_bgr, (640, 640))

                detected_object = first_detection(frame_bgr, yolo_detector)
                if detected_object is not None:
                    x1, y1, x2, y2 = detected_object
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    if ground_calibration_frames < 5:
                        base_y_bottom = y2
                        ground_calibration_frames += 1

                    lifted_in_image = (
                        base_y_bottom is not None and (base_y_bottom - y2) > 30
                    )
                    if lifted_in_image:
                        grasp_buffer = min(grasp_buffer + 1, 15)
                    else:
                        grasp_buffer = max(grasp_buffer - 1, 0)
                else:
                    grasp_buffer = max(grasp_buffer - 2, 0)

                cv_picked_status = grasp_buffer > 10
                status_text = "PICKED" if cv_picked_status else "NOT PICKED"
                color = (0, 255, 0) if cv_picked_status else (0, 0, 255)
                cv2.putText(
                    frame_bgr,
                    f"CV Status: {status_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

                sim_grasped = bool(info.get("is_grasped", False))
                sim_contact = bool(info.get("tip_object_contact", False))
                cv2.putText(
                    frame_bgr,
                    f"Sim Grasp: {sim_grasped}  Contact: {sim_contact}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(window_name, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return

                # Real-time pacing: one env step represents time_per_step
                # seconds of simulated time. Without this the demo plays as
                # fast as the CPU allows (~5-10x speed).
                elapsed = time.perf_counter() - step_started
                remaining = float(env.time_per_step) - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

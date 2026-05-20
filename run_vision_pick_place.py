import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# Try importing the required libraries
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics to use YOLO: pip install ultralytics")
    exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
except ImportError:
    print("Please install stable_baselines3 to load the RL model: pip install stable-baselines3")
    exit(1)

# Import the environment
from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv
from rex_tendon.configs.pick_place_config import PickPlaceConfig
from rex_tendon.training.rl.pick_place_training import load_pick_place_config

def main():
    parser = argparse.ArgumentParser(description="Vision-based Pick and Place with YOLO")
    parser.add_argument("--model_path", type=str, default="rex_results/pick_place/pick_place_20260331_052341/models/best_model/best_model.zip", help="Path to trained RL model")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLO model to use")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    # Load RL Model
    rl_model_path = Path(args.model_path)
    if not rl_model_path.exists():
        print(f"Error: RL model not found at {rl_model_path}. You might need to train it first or provide a valid path.")
        return
    
    print("Loading RL Model...")
    rl_model = PPO.load(str(rl_model_path))

    # Initialize Environment
    if args.config:
        config = load_pick_place_config(args.config)
    else:
        config = PickPlaceConfig()
    config.env.curriculum_enabled = False
    
    print("Initializing MuJoCo Environment...")
    env = TentaclePickPlaceEnv(config=config.env, render_mode="rgb_array")
    
    import mujoco
    env.image_size = (480, 480)
    if env.renderer:
        env.renderer.close()
    env.renderer = mujoco.Renderer(env.model, height=480, width=480)
    
    # Switch to the workspace camera for better view of lifting
    env.renderer.update_scene(env.data, camera="workspace_cam")

    # Wrap environment with normalization if stats file exists
    stats_path = rl_model_path.parent / "vecnormalize.pkl"
    if stats_path.exists():
        print(f"Loading normalization stats from {stats_path}...")
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(str(stats_path), venv)
        venv.training = False
        venv.norm_reward = False
        active_env = venv
    else:
        print("Warning: No normalization stats found. Model might not move correctly.")
        active_env = env

    # Load YOLO Model
    print("Loading YOLO Model...")
    yolo_detector = YOLO(args.yolo_model)

    obs = active_env.reset()
    
    # State tracking for CV logic
    cv_picked_status = False
    base_y_bottom = None
    lift_threshold = 30 # pixels
    ground_calibration_frames = 0
    
    window_name = "YOLO Vision - Pick & Place"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)

    print("\nStarting Simulation... Press 'q' to quit.")
    
    episodes = 20
    for ep in range(episodes):
        obs = active_env.reset()
        done = False
        base_y_bottom = None
        cv_picked_status = False
        grasp_buffer = 0  
        ground_calibration_frames = 0
        
        print(f"\n--- Starting Episode {ep+1} ---")
        
        while not done:
            # Predict action using RL policy
            action, _ = rl_model.predict(obs, deterministic=True)
            
            if isinstance(active_env, VecNormalize):
                obs, reward, dones, infos = active_env.step(action)
                done = dones[0]
                info = infos[0]
            else:
                obs, reward, terminated, truncated, info = active_env.step(action)
                done = terminated or truncated

            # Render frame
            env.renderer.update_scene(env.data, camera="workspace_cam")
            frame = env.render() 
            
            if frame is None:
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (640, 640))

            # Run YOLO detection with higher confidence threshold for speed
            results = yolo_detector.predict(frame_bgr, verbose=False, conf=0.25)
            
            detected_object = None
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0] # Just take the top one
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_object = (x1, y1, x2, y2)
            
            if detected_object:
                x1, y1, x2, y2 = detected_object
                # Thicker box for better visibility
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
                if ground_calibration_frames < 5:
                    base_y_bottom = y2
                    ground_calibration_frames += 1
                
                if (base_y_bottom - y2) > lift_threshold:
                    grasp_buffer = min(grasp_buffer + 1, 15)
                else:
                    grasp_buffer = max(grasp_buffer - 1, 0)
            else:
                # If object is lost, slowly drain the buffer
                grasp_buffer = max(grasp_buffer - 2, 0)
            
            cv_picked_status = grasp_buffer > 10
            
            # Text Overlays
            status_text = "PICKED" if cv_picked_status else "NOT PICKED"
            color = (0, 255, 0) if cv_picked_status else (0, 0, 255)
            cv2.putText(frame_bgr, f"CV Status: {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            sim_grasped = info.get("is_grasped", False)
            cv2.putText(frame_bgr, f"Sim Grasp: {sim_grasped}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow(window_name, frame_bgr)
            
            # Using 1ms instead of 60ms makes it much more responsive to 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                env.close()
                cv2.destroyAllWindows()
                return
                
    env.close()
    cv2.destroyAllWindows()
                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

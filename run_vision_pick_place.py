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
    
    # Switch to the corner camera for better view of lifting
    env.renderer.update_scene(env.data, camera="fixed_overview")

    # Load YOLO Model
    print("Loading YOLO Model...")
    yolo_detector = YOLO(args.yolo_model)

    obs, info = env.reset()
    
    # State tracking for CV logic
    cv_picked_status = False
    base_y_center = None
    lift_threshold = 20 # pixels
    
    window_name = "YOLO Vision - Pick & Place"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 640)

    print("\nStarting Simulation... Press 'q' to quit.")
    
    episodes = 20
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        base_y_center = None
        cv_picked_status = False
        
        while not done:
            # Predict action using RL policy
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render frame
            # Force update scene from a specific camera
            env.renderer.update_scene(env.data, camera="fixed_overview")
            frame = env.render() # Returns RGB array
            
            if frame is None:
                continue

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize the image to 640x640 so YOLO works better and text fits
            frame_bgr = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

            # --- COMPUTER VISION LOGIC (YOLO) ---
            # Run YOLO detection
            results = yolo_detector.predict(frame_bgr, verbose=False, conf=0.1)
            
            detected_object = None
            max_conf = 0.0
            
            # Find the most confident detection (the object we are picking)
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected_object = (x1, y1, x2, y2)
            
            # Draw detections and apply "Pick" intelligence
            if detected_object:
                x1, y1, x2, y2 = detected_object
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                y_center = (y1 + y2) / 2
                
                # Intelligence logic: If object moves up significantly, it's picked
                if base_y_center is None:
                    base_y_center = y_center
                
                # Y axis goes down in image coordinates. So smaller Y means higher.
                if (base_y_center - y_center) > lift_threshold:
                    cv_picked_status = True
                else:
                    cv_picked_status = False
            
            # Display Status
            status_text = "PICKED" if cv_picked_status else "NOT PICKED"
            color = (0, 255, 0) if cv_picked_status else (0, 0, 255)
            cv2.putText(frame_bgr, f"CV Status: {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show actual simulation info for comparison
            sim_grasped = info.get("is_grasped", False)
            cv2.putText(frame_bgr, f"Sim Grasp: {sim_grasped}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Display the frame
            cv2.imshow(window_name, frame_bgr)
            
            if cv2.waitKey(60) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return
                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

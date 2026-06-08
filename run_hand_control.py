#!/usr/bin/env python3
"""Launch script for hand-controlled tentacle simulation.

Usage:
    python run_hand_control.py                          # Direct control mode
    python run_hand_control.py --model path/to/model.zip  # RL-guided mode
    python run_hand_control.py --camera 1               # Different webcam
    python run_hand_control.py --smoothing 0.95          # Near-instant response
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from rex_tendon.control.hand_sim_controller import run_hand_controller


def main():
    parser = argparse.ArgumentParser(
        description="Direction-based hand control for the Rex Tendon robot simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hand Control Scheme (decoupled channels):
  Wrist position (L/R)             - Bend robot tip left / right
  Wrist position (U/D)             - Bend robot tip up / down
  Twist / roll hand                - Rotate the base (yaw) — true twist DOF
  Hand depth (closer/farther)      - Z extension: closer=extend, farther=retract
  Flat open palm                   - Return robot to neutral position
  Pinch THUMB + INDEX (hold)       - LOCK grasp on nearest object
  Pinch THUMB + MIDDLE finger      - UNLOCK / release grasped object

Keyboard:
  SPACE   - Unlock / release grasped object (backup)
  C       - Recalibrate neutral hand pose (do this first!)
  R       - Reset full simulation
  Q / ESC - Quit

Windows:
  MuJoCo viewer - Interactive 3-D scene (mouse to orbit/zoom)
  OpenCV window - Webcam feed + Robot Cam PiP + Table Corner Cam PiP

Modes:
  direct  - Wrist position maps directly to tendon lengths
  rl      - Wrist position is target; RL policy computes optimal action
        """,
    )

    parser.add_argument(
        "--xml",
        default="rex_assets/rex_simulation/pick_and_place_scene.xml",
        help="Path to MuJoCo XML scene file (default: pick_and_place_scene.xml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained RL model (.zip). Enables RL mode.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device ID (default: 0)",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "rl"],
        default=None,
        help="Control mode. Auto-detected: 'rl' if --model given, else 'direct'.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.90,
        help="Alpha for cursor smoothing: 0=max smooth, 1=instant/no-smooth. Default: 0.90",
    )

    args = parser.parse_args()

    # Auto-detect mode
    mode = args.mode
    if mode is None:
        mode = "rl" if args.model else "direct"

    if mode == "rl" and not args.model:
        print("Warning: RL mode selected but no --model provided. Using direct mode.")
        mode = "direct"

    print("=" * 60)
    print("  REX TENDON — DIRECTION HAND CONTROL")
    print("=" * 60)
    print(f"  Scene:     {args.xml}")
    print(f"  Mode:      {mode.upper()}")
    print(f"  Model:     {args.model or 'None (direct control)'}")
    print(f"  Camera:    {args.camera}")
    print(f"  Smoothing: {args.smoothing}")
    print("=" * 60)
    print("  Hand Controls:")
    print("    Wrist left/right       -> Bend tip L/R")
    print("    Wrist up/down          -> Bend tip U/D")
    print("    Twist / roll hand      -> Rotate base (yaw)")
    print("    Hand depth             -> Z extension (closer=extend)")
    print("    Flat open palm         -> Return to neutral position")
    print("    Pinch THUMB+INDEX      -> LOCK grasp (hold briefly)")
    print("    Pinch THUMB+MIDDLE     -> UNLOCK / release grasp")
    print("  Keyboard:")
    print("    SPACE              -> Release / unlock grasp (backup)")
    print("    'c'                -> Recalibrate neutral pose (do first!)")
    print("    'r'                -> Reset simulation")
    print("    'q' / ESC          -> Quit")
    print("=" * 60)

    run_hand_controller(
        xml_path=args.xml,
        model_path=args.model,
        camera_id=args.camera,
        mode=mode,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()



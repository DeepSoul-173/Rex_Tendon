#!/usr/bin/env python3
"""Voice/text-commanded tentacle simulation — control mode 3 of 3.

Natural-language commands drive the simulated arm through scripted movement
primitives. Parsing is rule-based (control/voice_commands.py): it needs NO
LLM, NO GPU, and no API key — suitable for the viva demo on any laptop.
Compound commands run as sequences, and objects are arranged into the arm's
reachable workspace at startup.

Usage:
    python run_voice_sim.py                       # typed commands + viewer
    python run_voice_sim.py --asr speech          # microphone (SpeechRecognition)
    python run_voice_sim.py --xml <scene.xml>     # different scene

Example commands:
    pick up the red cube
    take the red cube and put it on top of the purple
    put it in the corner            place it in the zone
    build a stack with all the cubes
    drop it and catch it            move left        go home
"""

from __future__ import annotations

import argparse
import queue
import sys

sys.path.insert(0, ".")

from rex_tendon.control.sim_primitives import SimArm, SimIntentExecutor
from rex_tendon.control.voice_commands import handle_text_sequence
from rex_tendon.control.voice_io import list_microphones, start_input_thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-commanded tentacle sim")
    parser.add_argument(
        "--xml",
        default="rex_assets/rex_simulation/pick_and_place_scene.xml",
        help="MuJoCo scene file",
    )
    parser.add_argument(
        "--asr",
        choices=["typed", "speech"],
        default="typed",
        help="Command input: typed stdin or microphone via SpeechRecognition",
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="Input device index for --asr speech (see --list-mics)",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphones and exit",
    )
    parser.add_argument(
        "--no-viewer", action="store_true", help="Headless (for testing)"
    )
    parser.add_argument(
        "--keep-layout",
        action="store_true",
        help="Keep the scene's default object layout instead of arranging "
        "everything into the reachable workspace",
    )
    args = parser.parse_args()

    if args.list_mics:
        mics = list_microphones()
        if not mics:
            print("No microphones found (is PyAudio installed?).")
        for idx, name in mics:
            print(f"  {idx}: {name}")
        return

    arm = SimArm(
        args.xml,
        viewer=not args.no_viewer,
        realtime=not args.no_viewer,
        arrange_objects=not args.keep_layout,
    )
    executor = SimIntentExecutor(arm)
    colors = arm.scene_colors()

    print("=" * 60)
    print("  REX TENDON — VOICE COMMAND CONTROL (rule-based, no LLM)")
    print("=" * 60)
    print(f"  Scene objects: "
          f"{', '.join(f'{o.color} {o.shape}' for o in arm.objects)}")
    print("  Try: 'take the red cube and put it on top of the purple',")
    print("       'build a stack with all the cubes', 'put it in the corner',")
    print("       'drop it and catch it', 'go home', 'quit'")
    print("=" * 60)

    commands: queue.Queue = queue.Queue()
    if args.asr == "typed":
        print("  MIC: OFF — typed mode (use --asr speech for the microphone)")
    start_input_thread(commands, mode=args.asr, mic_index=args.mic_index)

    try:
        while True:
            try:
                text = commands.get(timeout=0.05)
            except queue.Empty:
                arm.step(1)  # keep physics/viewer alive while idle
                continue
            if text.strip().lower() in {"quit", "exit"}:
                break
            for line in handle_text_sequence(text, executor, colors):
                print(line)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()
        print("Voice sim shut down.")


if __name__ == "__main__":
    main()

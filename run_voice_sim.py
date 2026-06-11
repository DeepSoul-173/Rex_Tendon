#!/usr/bin/env python3
"""Voice/text-commanded tentacle simulation — control mode 3 of 3.

Natural-language commands drive the simulated arm through scripted movement
primitives. Parsing is rule-based (control/voice_commands.py): it needs NO
LLM, NO GPU, and no API key — suitable for the viva demo on any laptop.
Speech input uses the SpeechRecognition package when requested; typed input
always works.

Usage:
    python run_voice_sim.py                       # typed commands + viewer
    python run_voice_sim.py --asr speech          # microphone (SpeechRecognition)
    python run_voice_sim.py --xml <scene.xml>     # different scene

Example commands:
    pick up the red cube          stack yellow on purple
    place it down                 move left
    let go                        go home
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

sys.path.insert(0, ".")

from rex_tendon.control.sim_primitives import SimArm, SimIntentExecutor
from rex_tendon.control.voice_commands import handle_text


def _typed_input_loop(commands: queue.Queue) -> None:
    while True:
        try:
            text = input("voice> ")
        except (EOFError, KeyboardInterrupt):
            commands.put("quit")
            return
        commands.put(text)
        if text.strip().lower() in {"quit", "exit"}:
            return


def _speech_input_loop(commands: queue.Queue) -> None:
    try:
        import speech_recognition as sr
    except ImportError:
        print("SpeechRecognition not installed — falling back to typed input.")
        _typed_input_loop(commands)
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
    print("Microphone ready — speak a command.")
    while True:
        with mic as source:
            audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            print(f"ASR error: {exc} — switching to typed input.")
            _typed_input_loop(commands)
            return
        print(f"Heard: {text}")
        commands.put(text)


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
        "--no-viewer", action="store_true", help="Headless (for testing)"
    )
    args = parser.parse_args()

    arm = SimArm(args.xml, viewer=not args.no_viewer, realtime=not args.no_viewer)
    executor = SimIntentExecutor(arm)
    colors = arm.scene_colors()

    print("=" * 60)
    print("  REX TENDON — VOICE COMMAND CONTROL (rule-based, no LLM)")
    print("=" * 60)
    print(f"  Scene objects: "
          f"{', '.join(f'{o.color} {o.shape}' for o in arm.objects)}")
    print("  Try: 'pick up the red cube', 'stack yellow on purple',")
    print("       'place it down', 'move left', 'let go', 'go home', 'quit'")
    print("=" * 60)

    commands: queue.Queue = queue.Queue()
    input_loop = _typed_input_loop if args.asr == "typed" else _speech_input_loop
    threading.Thread(target=input_loop, args=(commands,), daemon=True).start()

    try:
        while True:
            try:
                text = commands.get(timeout=0.05)
            except queue.Empty:
                arm.step(1)  # keep physics/viewer alive while idle
                continue
            if text.strip().lower() in {"quit", "exit"}:
                break
            result = handle_text(
                text,
                executor,
                available_colors=colors,
                holding_object=executor.holding_object,
            )
            print(result)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()
        print("Voice sim shut down.")


if __name__ == "__main__":
    main()

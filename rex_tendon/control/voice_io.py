"""Voice/text input threads feeding a command queue.

Shared by the standalone voice sim runner and the hand controller's voice
co-pilot. Typed input always works; microphone input uses the optional
SpeechRecognition package (Google Web Speech, no API key) and degrades to
typed input when unavailable. No LLM, no GPU.
"""

from __future__ import annotations

import queue
import threading


def typed_input_loop(commands: queue.Queue, prompt: str = "voice> ") -> None:
    """Read commands from stdin until 'quit'/'exit' or EOF."""
    while True:
        try:
            text = input(prompt)
        except (EOFError, KeyboardInterrupt):
            commands.put("quit")
            return
        commands.put(text)
        if text.strip().lower() in {"quit", "exit"}:
            return


def speech_input_loop(commands: queue.Queue) -> None:
    """Push recognized utterances onto the queue; typed fallback on failure."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("SpeechRecognition not installed — falling back to typed input.")
        typed_input_loop(commands)
        return

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
    except OSError:
        print("No microphone available — falling back to typed input.")
        typed_input_loop(commands)
        return

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
            typed_input_loop(commands)
            return
        print(f"Heard: {text}")
        commands.put(text)


def start_input_thread(
    commands: queue.Queue, mode: str = "typed", prompt: str = "voice> "
) -> threading.Thread:
    """Start the requested input loop as a daemon thread and return it."""
    if mode == "speech":
        target, args = speech_input_loop, (commands,)
    else:
        target, args = typed_input_loop, (commands, prompt)
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread

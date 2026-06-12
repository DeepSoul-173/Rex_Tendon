"""Voice/text input threads feeding a command queue.

Shared by the standalone voice sim runner and the hand controller's voice
co-pilot. Typed input always works; microphone input uses SpeechRecognition +
PyAudio (Google Web Speech, no API key) and degrades to typed input with an
EXPLICIT reason whenever anything in the audio stack fails — a silent mic is
worse than no mic. No LLM, no GPU.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

StatusCallback = Callable[[str], None]


def _report(status_cb: Optional[StatusCallback], message: str) -> None:
    print(message)
    if status_cb is not None:
        status_cb(message)


def list_microphones() -> list[tuple[int, str]]:
    """Enumerate available input devices: [(index, name), ...].

    Returns an empty list when the audio stack is unavailable.
    """
    try:
        import speech_recognition as sr

        return list(enumerate(sr.Microphone.list_microphone_names()))
    except Exception:
        return []


def typed_input_loop(
    commands: queue.Queue, prompt: str = "voice> "
) -> None:
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


def speech_input_loop(
    commands: queue.Queue,
    mic_index: Optional[int] = None,
    status_cb: Optional[StatusCallback] = None,
) -> None:
    """Push recognized utterances onto the queue.

    Every failure mode reports an explicit MIC: OFF reason and falls back to
    typed input, so the user always knows whether the microphone is live.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        _report(
            status_cb,
            "MIC: OFF — SpeechRecognition not installed "
            "(pip install SpeechRecognition pyaudio). Typed input active.",
        )
        typed_input_loop(commands)
        return

    try:
        device_names = sr.Microphone.list_microphone_names()
    except Exception as exc:
        _report(
            status_cb,
            f"MIC: OFF — audio stack unavailable ({type(exc).__name__}: {exc}). "
            "Install PyAudio: pip install pyaudio. Typed input active.",
        )
        typed_input_loop(commands)
        return
    if not device_names:
        _report(status_cb, "MIC: OFF — no input devices found. Typed input active.")
        typed_input_loop(commands)
        return

    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone(device_index=mic_index)
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
    except Exception as exc:
        _report(
            status_cb,
            f"MIC: OFF — could not open device "
            f"{mic_index if mic_index is not None else '(default)'} "
            f"({type(exc).__name__}: {exc}). Typed input active.",
        )
        typed_input_loop(commands)
        return

    device_label = (
        device_names[mic_index]
        if mic_index is not None and 0 <= mic_index < len(device_names)
        else "system default"
    )
    _report(status_cb, f"MIC: ON — listening on '{device_label}'. Speak a command.")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=5)
        except Exception as exc:
            _report(
                status_cb,
                f"MIC: OFF — capture failed ({type(exc).__name__}: {exc}). "
                "Typed input active.",
            )
            typed_input_loop(commands)
            return
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            continue  # silence / unintelligible — keep listening
        except sr.RequestError as exc:
            _report(
                status_cb,
                f"MIC: OFF — speech service unreachable ({exc}). "
                "Typed input active. (Google Web Speech needs internet.)",
            )
            typed_input_loop(commands)
            return
        _report(status_cb, f"MIC heard: {text}")
        commands.put(text)


def start_input_thread(
    commands: queue.Queue,
    mode: str = "typed",
    prompt: str = "voice> ",
    mic_index: Optional[int] = None,
    status_cb: Optional[StatusCallback] = None,
) -> threading.Thread:
    """Start the requested input loop as a daemon thread and return it."""
    if mode == "speech":
        target = speech_input_loop
        args: tuple = (commands, mic_index, status_cb)
    else:
        target = typed_input_loop
        args = (commands, prompt)
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread

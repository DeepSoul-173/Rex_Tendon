"""NemoClaw voice-command bridge for Rex Tendon.

The bridge intentionally keeps ASR optional. On development machines without a
microphone, SpeechRecognition, Whisper, or NeMo installed, it can still run in
typed or dry-run mode and exercise the command-to-robot wiring.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from rex_tendon.control.primitives import MotionBehavior
from rex_tendon.control.voice_commands import CommandAction, parse_intent

logger = logging.getLogger(__name__)

# Structured intents (control/voice_commands.py) that map onto hardware motion
# primitives. Object-level intents (stack/place/move) need the simulator's
# scene knowledge and are routed to run_voice_sim.py instead.
_INTENT_TO_BEHAVIOR = {
    CommandAction.PICK: MotionBehavior.GRAB,
    CommandAction.RELEASE: MotionBehavior.RELEASE,
    CommandAction.WAVE: MotionBehavior.SHAKE,
    CommandAction.STOP: MotionBehavior.NO,
}

_SIM_ONLY_ACTIONS = {
    CommandAction.STACK,
    CommandAction.PLACE,
    CommandAction.MOVE,
    CommandAction.NEUTRAL,
}


@dataclass(frozen=True)
class VoiceCommand:
    """Parsed user command."""

    action: MotionBehavior
    phrase: str


class NemoClawBridge:
    """Translate voice text into Rex Tendon robot actions."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        command_runner: Optional[Callable[[list[str]], subprocess.CompletedProcess]] = None,
    ) -> None:
        self.dry_run = dry_run
        self.command_runner = command_runner or self._run_subprocess

    @staticmethod
    def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse recognized text into a supported motion primitive."""
        phrase = text.strip().lower()
        if not phrase:
            return None

        command_map: list[tuple[tuple[str, ...], MotionBehavior]] = [
            (("grab", "grasp", "pick", "hold"), MotionBehavior.GRAB),
            (("release", "let go", "drop", "open"), MotionBehavior.RELEASE),
            (("yes", "agree", "correct"), MotionBehavior.YES),
            (("no", "stop", "wrong"), MotionBehavior.NO),
            (("wave", "hello", "hi"), MotionBehavior.SHAKE),
            (("circle", "celebrate", "happy"), MotionBehavior.CIRCLE),
            (("high five", "high-five"), MotionBehavior.HIGH_FIVE),
        ]

        for keywords, action in command_map:
            if any(keyword in phrase for keyword in keywords):
                return VoiceCommand(action=action, phrase=text)

        return None

    def execute(self, command: VoiceCommand) -> str:
        """Execute a parsed command using the Rex Tendon primitive CLI."""
        cmd = [sys.executable, "-m", "rex_tendon", "primitive", command.action.value]
        logger.info("NemoClaw command '%s' -> %s", command.phrase, command.action.value)

        if self.dry_run:
            return f"DRY RUN: {' '.join(cmd)}"

        if shutil.which(sys.executable) is None:
            raise RuntimeError(f"Python executable not found: {sys.executable}")

        try:
            result = self.command_runner(cmd)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or str(exc)
            logger.error("Robot action failed: %s", stderr)
            raise RuntimeError(stderr) from exc

        return result.stdout.strip() or f"Executed {command.action.value}"

    def handle_text(self, text: str) -> str:
        """Parse and execute one text command.

        Structured intents are tried first (shared grammar with the sim voice
        runner); the legacy keyword map remains as fallback for behaviors the
        intent grammar does not cover (yes/circle/high-five).
        """
        intent = parse_intent(text)
        if intent is not None:
            behavior = _INTENT_TO_BEHAVIOR.get(intent.action)
            if behavior is not None:
                return self.execute(VoiceCommand(action=behavior, phrase=text))
            if intent.action in _SIM_ONLY_ACTIONS:
                return (
                    f"'{intent.action.value}' is an object-level command — run "
                    "it in the simulator: python run_voice_sim.py"
                )

        command = self.parse_command(text)
        if command is None:
            return f"No supported NemoClaw command found in: {text!r}"
        return self.execute(command)


def recognize_once_speech_recognition(
    *, timeout: float = 5.0, phrase_time_limit: float = 6.0
) -> str:
    """Capture one utterance using the optional SpeechRecognition package."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise RuntimeError(
            "SpeechRecognition is not installed. Install it or run with --asr typed."
        ) from exc

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
    except OSError as exc:
        raise RuntimeError("Could not initialize microphone input.") from exc

    return recognizer.recognize_google(audio)


def recognize_once_typed() -> str:
    """Typed fallback for machines without a working microphone/ASR stack."""
    return input("NemoClaw command> ")


def bend_tentacle(pitch: float, yaw: float, extension: float, *, dry_run: bool = False) -> str:
    """Backward-compatible helper for older callers.

    The current Rex Tendon CLI exposes motion primitives rather than a direct
    bend subcommand, so this helper validates the request and reports it instead
    of invoking a nonexistent command.
    """
    message = f"bend request pitch={pitch:.3f}, yaw={yaw:.3f}, extension={extension:.3f}"
    if dry_run:
        return f"DRY RUN: {message}"
    raise NotImplementedError(
        "Direct bend_tentacle CLI control is not exposed. Use run_hand_control.py "
        "for continuous control or primitive commands for voice actions."
    )


def trigger_grasp(*, dry_run: bool = False) -> str:
    """Backward-compatible helper to trigger the grab primitive."""
    bridge = NemoClawBridge(dry_run=dry_run)
    return bridge.execute(VoiceCommand(MotionBehavior.GRAB, "trigger grasp"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = NemoClawBridge(dry_run=True)
    print(bridge.handle_text("grab the object"))

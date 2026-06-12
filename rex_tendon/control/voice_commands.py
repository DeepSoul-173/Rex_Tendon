"""Structured voice/text command schema, parser, and validation.

Sits between raw recognized text and execution. nemoclaw_bridge.py currently
keyword-matches straight to motion primitives; this module adds object-level
intents (pick the red cube, stack red on blue) with slots and validation, as
a separate layer the bridge and orchestrator can both call.

Deliberately imports nothing from control.primitives / hardware — parsing and
validating commands must not pull in the motor stack. Integration code maps
CommandAction values onto MotionBehavior / executors at the boundary.

Pipeline:  text -> parse_intent() -> VoiceIntent -> validate_intent() -> executor
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


class CommandAction(Enum):
    """High-level actions the robot supports."""

    PICK = "pick"
    PLACE = "place"
    STACK = "stack"
    STACK_ALL = "stack_all"  # composite: build a tower from every cube
    MOVE = "move"
    RELEASE = "release"
    STOP = "stop"
    NEUTRAL = "neutral"
    WAVE = "wave"


# Colors that can appear as object slots. Matches the simulation cube set
# (obj_cube, obj_cube_purple, obj_cube_yellow, obj_cube_extra_*); validation
# against the actually-loaded scene happens in validate_intent().
KNOWN_COLORS = ("red", "green", "blue", "purple", "yellow", "orange", "white", "gray")

KNOWN_DIRECTIONS = ("left", "right", "up", "down", "forward", "back")

# Named place locations the executor can resolve in the scene.
KNOWN_LOCATIONS = ("corner", "center", "middle", "zone", "target")

# Words that may refer to a graspable object.
_OBJECT_WORDS = r"(?:cube|die|dice|block|box|ball|sphere|object|one|it|something)"


@dataclass(frozen=True)
class VoiceIntent:
    """A parsed command. Slots are None when not present in the utterance."""

    action: CommandAction
    target_color: Optional[str] = None  # object to act on (None = held/nearest)
    destination_color: Optional[str] = None  # for STACK: base object
    direction: Optional[str] = None  # for MOVE
    location: Optional[str] = None  # for PLACE: named spot (corner/center/zone)
    raw_text: str = ""


# ── Parsing ───────────────────────────────────────────────────────────────────

_COLOR_RE = "|".join(KNOWN_COLORS)
_DIR_RE = "|".join(KNOWN_DIRECTIONS)

# Order matters: more specific patterns first (stack-all before stack, stack
# before pick — "stack the red cube on the blue cube" contains pick words too).
_RULES: list[tuple[CommandAction, re.Pattern[str]]] = [
    (
        CommandAction.STACK_ALL,
        re.compile(
            r"(?:\b(?:build|make)\b.*?\b(?:stack|tower|pile)\b"
            r"|\bstack\b.*?\b(?:all|every(?:thing)?)\b)"
        ),
    ),
    (
        CommandAction.STACK,
        # A placement verb anywhere + "<color?> on (top of) <color>". The
        # target is optional: "put it on top of the blue" stacks the HELD
        # object onto blue.
        re.compile(
            rf"(?=.*\b(?:stack|put|place|take|set|move|drop)\b)"
            rf"(?:.*?\b(?P<target>{_COLOR_RE})\b)?.*?"
            rf"\b(?:on\s+top\s+of|in\s+the\s+top\s+of|top\s+of|onto|on|over)\b"
            rf".*?\b(?P<dest>{_COLOR_RE})\b"
        ),
    ),
    (
        CommandAction.PICK,
        re.compile(
            rf"\b(?:pick|grab|grasp|take|get|lift|hold|catch|regrasp|re-grasp)\b"
            rf"(?:.*?\b(?P<target>{_COLOR_RE})\b)?.*?{_OBJECT_WORDS}"
        ),
    ),
    (
        CommandAction.PLACE,
        re.compile(
            rf"\b(?:place|put|set|drop)\b(?:.*?\b(?P<target>{_COLOR_RE})\b)?.*?"
            rf"\b(?:(?:in|at|to|near)\s+the\s+"
            rf"(?P<location>corner|center|middle|zone|target)"
            rf"|down|there|on\s+the\s+table)\b"
        ),
    ),
    (
        CommandAction.MOVE,
        re.compile(rf"\b(?:move|go|bend|lean)\b.*?\b(?P<direction>{_DIR_RE})\b"),
    ),
    (
        CommandAction.RELEASE,
        re.compile(r"\b(?:release|let go|drop|open)\b"),
    ),
    (
        CommandAction.STOP,
        re.compile(r"\b(?:stop|halt|freeze|hold still|wait)\b"),
    ),
    (
        CommandAction.NEUTRAL,
        re.compile(r"\b(?:neutral|home|reset|center|relax|straighten)\b"),
    ),
    (
        CommandAction.WAVE,
        re.compile(r"\b(?:wave|hello|hi there|say hi)\b"),
    ),
]


def parse_intent(text: str) -> Optional[VoiceIntent]:
    """Parse recognized text into a VoiceIntent, or None if not understood."""
    phrase = text.strip().lower()
    if not phrase:
        return None

    for action, pattern in _RULES:
        m = pattern.search(phrase)
        if m is None:
            continue
        groups = m.groupdict()
        return VoiceIntent(
            action=action,
            target_color=groups.get("target"),
            destination_color=groups.get("dest"),
            direction=groups.get("direction"),
            location=groups.get("location"),
            raw_text=text,
        )
    return None


# ── Validation ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""


def validate_intent(
    intent: VoiceIntent,
    available_colors: Optional[set[str]] = None,
    holding_object: bool = False,
) -> ValidationResult:
    """Check an intent against scene/robot state before execution.

    available_colors : colors of objects actually present in the scene
                       (None = skip object checks).
    holding_object   : whether the gripper currently holds something.
    """
    a = intent.action

    if a is CommandAction.STACK:
        if intent.destination_color is None:
            return ValidationResult(False, "stack needs a base color")
        if intent.target_color is None and not holding_object:
            return ValidationResult(
                False, "nothing is held — say which color to stack"
            )
        if intent.target_color == intent.destination_color:
            return ValidationResult(
                False, "cannot stack a color onto itself"
            )

    if a is CommandAction.PICK and holding_object:
        return ValidationResult(False, "already holding an object — release first")

    if a in (CommandAction.PLACE, CommandAction.RELEASE) and not holding_object:
        if a is CommandAction.PLACE:
            return ValidationResult(False, "nothing is held — pick first")

    if available_colors is not None:
        for color in (intent.target_color, intent.destination_color):
            if color is not None and color not in available_colors:
                return ValidationResult(
                    False, f"no {color} object in the scene"
                )

    return ValidationResult(True)


# ── Execution boundary ────────────────────────────────────────────────────────


class IntentExecutor(Protocol):
    """Anything that can carry out a validated intent.

    Implementations live at the integration layer (nemoclaw_bridge,
    orchestrator, sim controller) — not here.
    """

    def execute(self, intent: VoiceIntent) -> str:
        """Perform the intent; return a short human-readable status."""
        ...


class DryRunExecutor:
    """Echoes what would be executed. Default for tests and --dry-run."""

    def execute(self, intent: VoiceIntent) -> str:
        slots = []
        if intent.target_color:
            slots.append(f"target={intent.target_color}")
        if intent.destination_color:
            slots.append(f"dest={intent.destination_color}")
        if intent.direction:
            slots.append(f"dir={intent.direction}")
        if intent.location:
            slots.append(f"loc={intent.location}")
        detail = f" ({', '.join(slots)})" if slots else ""
        return f"DRY RUN: {intent.action.value}{detail}"


def handle_text(
    text: str,
    executor: IntentExecutor,
    available_colors: Optional[set[str]] = None,
    holding_object: bool = False,
) -> str:
    """Full pipeline: parse -> validate -> execute, with readable errors."""
    intent = parse_intent(text)
    if intent is None:
        return f"Not understood: '{text.strip()}'"
    verdict = validate_intent(intent, available_colors, holding_object)
    if not verdict.ok:
        return f"Rejected ({intent.action.value}): {verdict.reason}"
    return executor.execute(intent)


_SEQUENCE_SPLIT = re.compile(r"\s+(?:and\s+then|then|and)\s+")


def handle_text_sequence(
    text: str,
    executor: IntentExecutor,
    available_colors: Optional[set[str]] = None,
) -> list[str]:
    """Execute a compound command: 'take the red cube and put it on the blue'.

    Segments split on and/then run in order; the gripper state is re-queried
    from the executor between segments (so 'drop it and catch it' works).
    Stops at the first segment that fails to parse or validate.
    """
    results: list[str] = []
    for segment in _SEQUENCE_SPLIT.split(text.strip()):
        if not segment.strip():
            continue
        holding = bool(getattr(executor, "holding_object", False))
        out = handle_text(segment, executor, available_colors, holding)
        results.append(out)
        if out.startswith(("Not understood", "Rejected")):
            break
    return results

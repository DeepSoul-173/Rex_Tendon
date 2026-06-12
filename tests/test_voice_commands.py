"""Unit tests for rex_tendon.control.voice_commands (no robot imports)."""

import pytest

from rex_tendon.control.voice_commands import (
    CommandAction,
    DryRunExecutor,
    VoiceIntent,
    handle_text,
    handle_text_sequence,
    parse_intent,
    validate_intent,
)

SCENE = {"red", "green", "blue", "purple", "yellow"}


# ── Parsing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,action,target,dest,direction",
    [
        ("pick up the red cube", CommandAction.PICK, "red", None, None),
        ("grab the blue die", CommandAction.PICK, "blue", None, None),
        ("take that green block", CommandAction.PICK, "green", None, None),
        ("pick up the cube", CommandAction.PICK, None, None, None),
        ("stack the red cube on the blue cube", CommandAction.STACK, "red", "blue", None),
        ("put yellow on top of purple", CommandAction.STACK, "yellow", "purple", None),
        ("place the cube down", CommandAction.PLACE, None, None, None),
        ("put it down", CommandAction.PLACE, None, None, None),
        ("move left", CommandAction.MOVE, None, None, "left"),
        ("go forward a bit", CommandAction.MOVE, None, None, "forward"),
        ("let go", CommandAction.RELEASE, None, None, None),
        ("drop it", CommandAction.RELEASE, None, None, None),
        ("stop", CommandAction.STOP, None, None, None),
        ("go home", CommandAction.NEUTRAL, None, None, None),
        ("reset position", CommandAction.NEUTRAL, None, None, None),
        ("wave", CommandAction.WAVE, None, None, None),
    ],
)
def test_parse_examples(text, action, target, dest, direction):
    intent = parse_intent(text)
    assert intent is not None, f"failed to parse: {text}"
    assert intent.action is action
    assert intent.target_color == target
    assert intent.destination_color == dest
    assert intent.direction == direction


def test_parse_case_and_whitespace():
    intent = parse_intent("  PICK UP THE RED CUBE  ")
    assert intent is not None and intent.action is CommandAction.PICK
    assert intent.target_color == "red"


@pytest.mark.parametrize("text", ["", "   ", "what is the weather", "banana"])
def test_parse_rejects_nonsense(text):
    assert parse_intent(text) is None


def test_parse_keeps_raw_text():
    intent = parse_intent("Grab the blue cube")
    assert intent is not None and intent.raw_text == "Grab the blue cube"


@pytest.mark.parametrize(
    "text,action,target,dest,location",
    [
        # The user's own phrasings:
        ("take the red cube and put it in the top of the blue",
         CommandAction.STACK, "red", "blue", None),
        ("build a cube stack with all the cubes",
         CommandAction.STACK_ALL, None, None, None),
        ("stack all the cubes", CommandAction.STACK_ALL, None, None, None),
        ("grasp a ball or something", CommandAction.PICK, None, None, None),
        # Held-object stacking and named locations:
        ("put it on top of the blue", CommandAction.STACK, None, "blue", None),
        ("put it in the corner", CommandAction.PLACE, None, None, "corner"),
        ("place it at the center", CommandAction.PLACE, None, None, "center"),
        ("drop it in the zone", CommandAction.PLACE, None, None, "zone"),
        ("catch it", CommandAction.PICK, None, None, None),
        ("drop it", CommandAction.RELEASE, None, None, None),
    ],
)
def test_parse_rich_examples(text, action, target, dest, location):
    intent = parse_intent(text)
    assert intent is not None, f"failed to parse: {text}"
    assert intent.action is action
    assert intent.target_color == target
    assert intent.destination_color == dest
    assert intent.location == location


class _StatefulDryRun(DryRunExecutor):
    """DryRun that tracks gripper state, for sequence tests."""

    def __init__(self, holding: bool = False):
        self.holding_object = holding

    def execute(self, intent):
        if intent.action is CommandAction.PICK:
            self.holding_object = True
        elif intent.action in (
            CommandAction.RELEASE,
            CommandAction.PLACE,
            CommandAction.STACK,
        ):
            self.holding_object = False
        return super().execute(intent)


def test_sequence_pick_then_stack_held():
    ex = _StatefulDryRun()
    out = handle_text_sequence(
        "take the red cube and put it on top of the purple", ex, SCENE
    )
    assert out == ["DRY RUN: pick (target=red)", "DRY RUN: stack (dest=purple)"]


def test_sequence_drop_and_catch():
    ex = _StatefulDryRun(holding=True)
    out = handle_text_sequence("drop it and catch it", ex, SCENE)
    assert out == ["DRY RUN: release", "DRY RUN: pick"]


def test_sequence_stops_on_rejection():
    ex = _StatefulDryRun(holding=False)
    out = handle_text_sequence("put it on top of the purple and wave", ex, SCENE)
    assert len(out) == 1
    assert out[0].startswith("Rejected (stack)")


# ── Validation ────────────────────────────────────────────────────────────────


def test_validate_stack_needs_both_colors():
    intent = VoiceIntent(action=CommandAction.STACK, target_color="red")
    assert not validate_intent(intent).ok


def test_validate_stack_rejects_same_color():
    intent = VoiceIntent(
        action=CommandAction.STACK, target_color="red", destination_color="red"
    )
    assert not validate_intent(intent).ok


def test_validate_unknown_color_rejected():
    intent = VoiceIntent(action=CommandAction.PICK, target_color="orange")
    verdict = validate_intent(intent, available_colors=SCENE)
    assert not verdict.ok
    assert "orange" in verdict.reason


def test_validate_pick_while_holding_rejected():
    intent = VoiceIntent(action=CommandAction.PICK, target_color="red")
    assert not validate_intent(intent, SCENE, holding_object=True).ok


def test_validate_place_without_holding_rejected():
    intent = VoiceIntent(action=CommandAction.PLACE)
    assert not validate_intent(intent, SCENE, holding_object=False).ok


def test_validate_good_intents_pass():
    pick = VoiceIntent(action=CommandAction.PICK, target_color="red")
    assert validate_intent(pick, SCENE).ok
    stack = VoiceIntent(
        action=CommandAction.STACK, target_color="red", destination_color="blue"
    )
    assert validate_intent(stack, SCENE).ok
    stop = VoiceIntent(action=CommandAction.STOP)
    assert validate_intent(stop, SCENE, holding_object=True).ok


# ── Pipeline ──────────────────────────────────────────────────────────────────


def test_handle_text_executes_valid_command():
    out = handle_text("pick up the red cube", DryRunExecutor(), SCENE)
    assert out == "DRY RUN: pick (target=red)"


def test_handle_text_stack_slots():
    out = handle_text("stack red on blue", DryRunExecutor(), SCENE)
    assert out == "DRY RUN: stack (target=red, dest=blue)"


def test_handle_text_reports_not_understood():
    out = handle_text("sing me a song", DryRunExecutor(), SCENE)
    assert out.startswith("Not understood")


def test_handle_text_reports_rejection():
    out = handle_text(
        "pick up the red cube", DryRunExecutor(), SCENE, holding_object=True
    )
    assert out.startswith("Rejected (pick)")

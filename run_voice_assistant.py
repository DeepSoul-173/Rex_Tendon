"""Run the Rex Tendon voice assistant.

Two backends are supported:
- openai: full realtime orchestrator with microphone, OpenAI Realtime, vision cues.
- local: lightweight NemoClaw command parser for offline microphone or typed input.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from nemoclaw_bridge import (
    NemoClawBridge,
    recognize_once_speech_recognition,
    recognize_once_typed,
)

try:
    from rich.console import Console
except ImportError:

    class Console:  # type: ignore[no-redef]
        def print(self, *args: object, **_: object) -> None:
            print(*args)

        def rule(self, title: str) -> None:
            print(f"\n{title}\n")


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("VoiceAssistant")
console = Console()


async def run_openai_backend(config_path: str | None) -> None:
    """Run the full realtime orchestrator backend."""
    from rex_tendon.configs.loaders import get_orchestrator_config
    from rex_tendon.orchestrator.orchestrator import OrchestratorApp

    config = get_orchestrator_config(config_path)
    if not config.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it or run with --backend local."
        )

    app = OrchestratorApp(orchestrator_config=config)
    await app.start()


def run_local_backend(*, asr: str, dry_run: bool, once: bool) -> None:
    """Run the offline NemoClaw command loop."""
    bridge = NemoClawBridge(dry_run=dry_run)
    recognizer = recognize_once_typed
    if asr == "speech_recognition":
        recognizer = recognize_once_speech_recognition

    console.print("[cyan]NemoClaw local voice bridge ready.[/cyan]")
    if dry_run:
        console.print(
            "[yellow]Dry-run mode: commands are parsed but motors are not moved.[/yellow]"
        )

    while True:
        try:
            text = recognizer()
            if text.strip().lower() in {"quit", "exit"}:
                return
            console.print(f"[dim]Heard:[/dim] {text}")
            console.print(bridge.handle_text(text))
        except KeyboardInterrupt:
            return
        except Exception as exc:
            logger.error("Voice loop error: %s", exc)

        if once:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rex Tendon voice assistant")
    parser.add_argument(
        "--backend",
        choices=("local", "openai"),
        default="local",
        help="Voice backend to run. Use openai for the full realtime orchestrator.",
    )
    parser.add_argument(
        "--asr",
        choices=("typed", "speech_recognition"),
        default="typed",
        help="Input method for --backend local.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional orchestrator YAML config for --backend openai.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse local commands without sending motor commands.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single local command and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console.rule("[bold cyan]Rex Tendon Voice Assistant")

    if args.backend == "local":
        run_local_backend(asr=args.asr, dry_run=args.dry_run, once=args.once)
        return

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run_openai_backend(args.config))
    except KeyboardInterrupt:
        console.print("[yellow]Shutting down voice assistant...[/yellow]")
    except Exception as exc:
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        raise


if __name__ == "__main__":
    main()

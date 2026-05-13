import asyncio
import logging
from rich.console import Console
import sys
import os

from rex_tendon.orchestrator.orchestrator import OrchestratorApp
from rex_tendon.configs.orchestrator import OrchestratorConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("VoiceAssistant")
console = Console()


# --- PASTE YOUR API KEY HERE IF NOT USING ENV VARIABLES ---
API_KEY = ""
# --------------------------------------------------------

async def main():
    console.rule("[bold cyan]Rex Tendon Voice Assistant")
    console.print("[dim]Starting OpenAI Realtime Voice Interface...[/dim]")

    if API_KEY:
        os.environ["OPENAI_API_KEY"] = API_KEY

    if not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[bold red]ERROR:[/bold red] OPENAI_API_KEY environment variable is not set."
        )
        console.print("Please set your API key before running the Voice component.")
        sys.exit(1)

    # Initialize app with default configurations
    app = OrchestratorApp()

    try:
        await app.start()
    except KeyboardInterrupt:
        console.print("[yellow]Shutting down voice assistant...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")


if __name__ == "__main__":
    try:
        # For Windows compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

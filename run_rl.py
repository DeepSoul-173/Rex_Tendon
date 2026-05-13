import typer
from rex_tendon.training.rl.pick_place_training import app as pick_place_app

if __name__ == "__main__":
    app = typer.Typer()
    app.add_typer(pick_place_app, name="pick-place")
    app()



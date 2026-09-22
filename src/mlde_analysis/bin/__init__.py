from dotenv import load_dotenv
import typer  # add noqa:E402

load_dotenv()  # take environment variables from .env.

from . import hist2d, stats  # noqa:E402

app = typer.Typer()
app.add_typer(hist2d.app, name="hist2d")
app.add_typer(stats.app, name="stats")

if __name__ == "__main__":
    app()

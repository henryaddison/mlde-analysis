import typer

from . import hist2d

app = typer.Typer()
app.add_typer(hist2d.app, name="hist2d")


if __name__ == "__main__":
    app()

from pathlib import Path
import xarray as xr
import shutil
import os

import typer

app = typer.Typer()


@app.command()
def main(
    filepath: Path,
):
    rechunked_filepath = Path(
        str(filepath).replace("predictions", "predictions-rechunked")
    )

    typer.echo(f"Rechunking {filepath} to {rechunked_filepath}")

    ds = xr.open_dataset(filepath, chunks={})
    # remove existing chunking info to avoid conflicts
    del ds["pr"].encoding["chunks"]
    ds["pr"] = ds["pr"].chunk(
        {
            "ensemble_member": 1,
            "time": 90,
            "frame": 24,
            "grid_longitude": 64,
            "grid_latitude": 64,
        }
    )

    ds.to_zarr(rechunked_filepath)

    typer.echo(f"Replacing {filepath} with {rechunked_filepath}")
    shutil.rmtree(f"{filepath}.bak", ignore_errors=True)
    shutil.move(filepath, f"{filepath}.bak")
    os.replace(rechunked_filepath, filepath)
    xr.open_dataset(filepath, chunks={}).close()


if __name__ == "__main__":
    app()

import logging
import pandas as pd
from pathlib import Path
import typer
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def main(samples_path: Path):
    """
    Combine the time and frame dimensions of a set of samples into a single time dimension.
    """
    logger.info(f"Combining time and frame dimensions for {samples_path}...")
    ds = xr.load_dataset(samples_path)
    ds = ds.stack(valid_time=("time", "frame"))
    ds = ds.assign_coords(
        time_and_frame=ds.time
        + pd.to_timedelta(ds.frame, unit="h").to_pytimedelta()
        + pd.to_timedelta(30, unit="min").to_pytimedelta()
    )
    ds = (
        ds.swap_dims({"valid_time": "time_and_frame"})
        .drop_vars(["time", "frame", "valid_time"])
        .rename({"time_and_frame": "time"})
    )

    logger.info(f"Saving corrected dataset to {samples_path}...")
    ds.to_zarr(samples_path, mode="w")


if __name__ == "__main__":
    app()

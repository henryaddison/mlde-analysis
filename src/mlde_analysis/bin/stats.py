import logging
from pathlib import Path
import typer
import xarray as xr
from mlde_analysis import stats


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def store(sample_run_path: Path, eval_vars: list[str]):
    """Store evaluation statistics for a run of samples split over simualation ensemble members."""

    ds = xr.concat(
        [
            xr.open_dataset(fp, chunks={})
            for fp in sample_run_path.glob("*/predictions.zarr")
        ],
        dim="ensemble_member",
    )

    samples_stats = stats.from_dataset(ds, (0, 200), eval_vars).compute()

    samples_stats.to_zarr(sample_run_path / f"eval_stats.zarr", mode="w")

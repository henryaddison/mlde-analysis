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
def predictions(prediction_run_path: Path, eval_vars: list[str]):
    """Store evaluation statistics for a run of predictions split over simualation ensemble members."""

    logger.info(
        f"Computing prediction run evaluation statistics for {prediction_run_path}..."
    )
    ds = xr.concat(
        [
            xr.open_dataset(fp, chunks={})
            for fp in prediction_run_path.glob("*/predictions.zarr")
        ],
        dim="ensemble_member",
        data_vars="minimal",
        coords="minimal",
        join="exact",
    )

    samples_stats = stats.from_dataset(ds, (0, 200), eval_vars).compute()

    samples_stats.to_zarr(prediction_run_path / f"eval_stats.zarr", mode="w")


@app.command()
def dataset(dataset_split_path: Path, eval_vars: list[str]):
    """Store evaluation statistics for a dataset of samples split over simulation ensemble members."""
    logger.info(f"Computing dataset evaluation statistics for {dataset_split_path}...")
    ds = xr.load_dataset(dataset_split_path / "predictands.zarr")

    samples_stats = stats.from_dataset(ds, (0, 200), eval_vars).compute()

    samples_stats.to_zarr(dataset_split_path / f"eval_stats.zarr", mode="w")

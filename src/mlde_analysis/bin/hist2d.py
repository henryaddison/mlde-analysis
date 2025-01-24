from dotenv import load_dotenv
import logging
from pathlib import Path
import typer
import numpy as np

from papermill.iorw import read_yaml_file
from ..data import prep_eval_data
from ..display import VAR_RANGES
from ..mv_distribution import compute_hist2d
from .. import default_params

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()

load_dotenv()  # take environment variables from .env.


@app.callback()
def callback():
    pass


def _load_params(params_file: Path):
    params = {
        key: value
        for key, value in vars(default_params).items()
        if not (key.startswith("__") or key.startswith("_"))
    }
    if params_file is not None:
        params.update(read_yaml_file(str(params_file)))
    return params


def _load_data(params):
    return prep_eval_data(
        params["sample_configs"],
        params["dataset_configs"],
        params["derived_variables_config"],
        params["eval_vars"],
        params["split"],
        ensemble_members=params["ensemble_members"],
        samples_per_run=params["samples_per_run"],
    )


@app.command()
def seasonal(
    out_path: Path, params_file: Path = None, xvar: str = "pr", yvar: str = "tmean150cm"
):
    logger.info(f"Loading data for {params_file}...")
    params = _load_params(params_file)
    eval_ds, _ = _load_data(params)

    logger.info(
        f"Computing seasonal 2D frequency density histograms for {xvar} and {yvar}..."
    )

    def extract_and_compute_hist2d(ds, xvar, yvar, xbins, ybins):
        x_pred = ds[f"pred_{xvar}"]
        y_pred = ds[f"pred_{yvar}"]
        x_target = ds[f"target_{xvar}"]
        y_target = ds[f"target_{yvar}"]

        return compute_hist2d(
            x_pred, y_pred, x_target, y_target, xbins=xbins, ybins=ybins
        )

    ds = eval_ds["CPM"]
    xbins = np.histogram_bin_edges([], bins=50, range=VAR_RANGES[xvar])
    ybins = np.histogram_bin_edges([], bins=50, range=VAR_RANGES[yvar])
    hist2d_ds = ds.groupby("time.season").map(
        extract_and_compute_hist2d, xvar=xvar, yvar=yvar, xbins=xbins, ybins=ybins
    )

    # compress data when saving
    for var in list(hist2d_ds.data_vars):
        hist2d_ds[var].encoding.update(zlib=True, complevel=5)

    logger.info(f"Saving result to {out_path}...")
    hist2d_ds.to_netcdf(out_path)

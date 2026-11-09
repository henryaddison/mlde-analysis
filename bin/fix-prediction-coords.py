import cf_xarray  # noqa: F401
from dotenv import load_dotenv
import logging
from pathlib import Path
import shutil
import typer
import xarray as xr

load_dotenv()

from mlde_utils import FurflexDatasetMetadata  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def main(samples_path: Path, dataset: str):
    """
    Fix the attributes and spatial coordinates and grid mapping of a set of samples.
    """
    logger.info(f"Fixing coords for {samples_path}...")
    samples_ds = xr.load_dataset(samples_path)
    template_ds = xr.load_dataset(
        FurflexDatasetMetadata(dataset).path() / "template.nc"
    )

    logger.info(f"Moving existing samples to backup location")
    shutil.move(samples_path, f"{samples_path}.bak")

    template_grid_mapping = template_ds.cf.grid_mappings[0]
    template_x_coord = template_ds.cf.coords["X"]
    template_y_coord = template_ds.cf.coords["Y"]

    samples_ds = samples_ds.drop_vars(samples_ds.cf.grid_mappings[0].name)
    samples_ds[template_grid_mapping.name] = template_grid_mapping.array

    samples_ds = samples_ds.rename(
        {
            "grid_latitude": template_y_coord.name,
            "grid_longitude": template_x_coord.name,
        }
    )
    # assign the spatial coordinates to sample spatial dimensions to match the target dataset
    samples_ds = samples_ds.assign_coords(
        {
            template_y_coord.name: template_y_coord,
            template_x_coord.name: template_x_coord,
        }
    )

    samples_ds["pr"] = samples_ds["pr"].assign_attrs(
        template_ds["pr"].attrs | {"standard_name": "pred_pr"}
    )

    logger.info(f"Saving corrected dataset to {samples_path}...")
    samples_ds.to_zarr(samples_path, mode="w")
    logger.info(f"Removing backup of old samples now new ones are written")
    shutil.rmtree(f"{samples_path}.bak", ignore_errors=True)


if __name__ == "__main__":
    app()

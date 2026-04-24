import glob
from mlde_utils import DATASETS_PATH
import os
from pathlib import Path
import xarray as xr


class FurflexDatasetMetadata:
    def __init__(self, name, base_dir=DATASETS_PATH):
        self.name = name
        self.base_dir = base_dir

    def __str__(self):
        return f"DatasetMetadata({self.path()})"

    def path(self):
        return Path(self.base_dir, self.name)

    def splits(self):
        return map(
            lambda f: os.path.splitext(f)[0],
            glob.glob("*", root_dir=str(self.path())),
        )

    def split_path(self, split):
        return self.path() / split

    def predictands_split_path(self, split):
        return self.split_path(split) / "predictands.zarr"

    # def config_path(self) -> Path:
    #     return self.path() / "ds-config.yml"

    # def config(self) -> dict:
    #     with open(self.config_path(), "r") as f:
    #         return yaml.safe_load(f)

    # def ensemble_members(self) -> list[str]:
    #     return self.config()["ensemble_members"]


def open_dataset_split(dataset_name, split, ensemble_members="all"):
    if ensemble_members == "all":
        ds = xr.open_dataset(
            FurflexDatasetMetadata(dataset_name).predictands_split_path(split)
        )
    else:
        ds = xr.open_dataset(
            FurflexDatasetMetadata(dataset_name).predictands_split_path(split)
        ).sel(ensemble_member=ensemble_members)

    return ds

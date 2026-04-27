import glob
import importlib
from mlde_utils import DATASETS_PATH, TIME_PERIODS
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

from . import display

WORKDIRS_PATH = Path(os.getenv("WORKDIRS_PATH"))


class FurflexEmulatorOutputMetadata:
    def __init__(self, fq_run_id: str, base_dir: Path):
        self.base_dir = base_dir
        self.fq_run_id = fq_run_id

    def workdir_path(self) -> Path:
        """
        Returns the path to the emulator output for the given run ID.
        """
        return Path(self.base_dir, self.fq_run_id)

    def __str__(self) -> str:
        return f"FurflexEmulatorOutputMetadata(path={self.workdir_path()})"

    def samples_path(
        self,
        checkpoint: str,
        input_xfm: str,
        dataset: str,
        split: str,
        ensemble_member: str,
        config_hash: str,  # missing from older outputs, use None in that case
    ) -> Path:
        """
        Returns the path to the samples for the given parameters.
        """
        path = (
            self.workdir_path()
            / "samples"
            / checkpoint
            / dataset
            # / input_xfm
            / split
            / ensemble_member
        )
        if config_hash is not None:
            path = path / config_hash
        return path

    def samples_glob(self, *args, **kwargs) -> list[Path]:
        """
        Returns a list of prediction files for the given parameters
        """
        return self.samples_path(*args, **kwargs).glob("*/predictions.zarr")


class FurflexDatasetMetadata:
    def __init__(self, name, base_dir=DATASETS_PATH):
        self.name = name
        self.base_dir = base_dir

    def __str__(self):
        return f"FurflexDatasetMetadata({self.path()})"

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


def _exclude_days(ds, exclude_days):
    """
    Exclude a margin of n days at the start and end of each season to avoid risks of data leakage from training set via autocorrelation.
    """
    if exclude_days > 0:
        # WARNING: this exclusion logic is designed for random season split strategy
        # TODO: make this exclusion depend on the split strategy
        doy_whitelist = np.concat(
            [
                (
                    np.arange(
                        60 + i * 90 + exclude_days, 60 + (i + 1) * 90 - exclude_days
                    )
                    % 360
                )
                + 1
                for i in range(4)
            ]
        )
        ds = ds.sel(time=ds.time.dt.dayofyear.isin(doy_whitelist))
    return ds


def prep_eval_data(
    sample_configs,
    dataset_configs,
    derived_var_configs,
    eval_vars,
    split,
    exclude_days,
    ensemble_members,
    samples_per_run,
    coarsen_time=None,
):
    models = {
        source: dict(
            sorted(
                {
                    run_config["label"]: {"order": -1, "CCS": False, "source": source}
                    | run_config
                    for run_config in data_configs
                }.items(),
                key=lambda x: x[1]["order"],
            )
        )
        for source, data_configs in sample_configs.items()
    }

    merged_ds = {}
    for source, sample_config in sample_configs.items():
        dataset_ds = open_dataset_split(
            dataset_configs[source], split, ensemble_members
        )
        dataset_ds = dataset_ds.rename(
            {var: f"target_{var}" for var in eval_vars if var in dataset_ds.data_vars}
        )
        dataset_ds = _exclude_days(dataset_ds, exclude_days)

        for var, attrs in display.ATTRS.items():
            if var in dataset_ds.data_vars:
                dataset_ds[var] = dataset_ds[var].assign_attrs(attrs)

        samples_ds = open_concat_sample_datasets(
            sample_config,
            split=split,
            ensemble_members=ensemble_members,
            samples_per_run=samples_per_run,
        )
        samples_ds = samples_ds.rename(
            {var: f"pred_{var}" for var in eval_vars if var in samples_ds.data_vars}
        )

        for var, attrs in display.ATTRS.items():
            pvarname = f"pred_{var}"
            if pvarname in samples_ds.data_vars:
                samples_ds[pvarname] = samples_ds[pvarname].assign_attrs(
                    dataset_ds[f"target_{var}"].attrs | attrs
                )

        samples_ds = samples_ds.assign_coords(
            grid_latitude=dataset_ds["grid_latitude"].copy(),
            grid_longitude=dataset_ds["grid_longitude"].copy(),
        )

        ds = xr.merge([samples_ds, dataset_ds], join="inner", compat="override")
        assert len(dataset_ds["time"]) == len(ds["time"]), (
            f"Different time length for dataset before and after merging with samples: "
            f"{len(ds['time'])} != {len(dataset_ds['time'])}. "
            "Perhaps samples do not cover the time period of the dataset."
        )

        if coarsen_time is not None:
            # ds = ds.assign_coords(date=ds.time.dt.floor("D")).groupby("date").mean().rename(date="time")
            # ds = ds.coarsen(time=24).mean(keep_attrs=True)
            ds = (
                ds.drop_vars(
                    [
                        "time_bnds",
                        "month_number",  # TODO: these should already be removed from datasets
                        "year",  # TODO: these should already be removed from datasets
                        "yyyymmddhh",  # TODO: these should already be removed from datasets
                    ],
                    errors="ignore",
                )
                .coarsen(time=coarsen_time)
                .sum(keep_attrs=True)
            )

        ds = attach_eval_coords(ds)

        ds = attach_derived_variables(ds, derived_var_configs)

        merged_ds[source] = ds

    return merged_ds, models


def open_samples_ds(
    run_name,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    ensemble_members,
    num_samples,
    deterministic,
    config_hash=None,
):
    eo_meta = FurflexEmulatorOutputMetadata(fq_run_id=run_name, base_dir=WORKDIRS_PATH)
    per_em_datasets = []
    for ensemble_member in ensemble_members:
        samples_dir = eo_meta.samples_path(
            checkpoint=checkpoint_id,
            input_xfm=input_xfm_key,
            dataset=dataset_name,
            split=split,
            ensemble_member=ensemble_member,
            config_hash=config_hash,
        )
        sample_files_list = list(
            eo_meta.samples_glob(
                checkpoint=checkpoint_id,
                input_xfm=input_xfm_key,
                dataset=dataset_name,
                split=split,
                ensemble_member=ensemble_member,
                config_hash=config_hash,
            )
        )
        if len(sample_files_list) == 0:
            raise RuntimeError(f"{samples_dir} has no sample files")

        if deterministic:
            em_ds = xr.open_dataset(sample_files_list[0])
        else:
            sample_files_list = sample_files_list[:num_samples]
            if len(sample_files_list) < num_samples:
                raise RuntimeError(
                    f"{samples_dir} does not have {num_samples} sample files"
                )
            em_ds = xr.concat(
                [
                    xr.open_dataset(sample_filepath)
                    for sample_filepath in sample_files_list
                ],
                dim="sample_id",
            ).isel(sample_id=range(num_samples))

        em_ds = em_ds.stack(valid_time=("time", "frame"))
        em_ds = em_ds.assign_coords(
            time_and_frame=em_ds.time
            + pd.to_timedelta(em_ds.frame, unit="h").to_pytimedelta()
            + pd.to_timedelta(30, unit="min").to_pytimedelta()
        )
        em_ds = (
            em_ds.swap_dims({"valid_time": "time_and_frame"})
            .drop_vars(["time", "frame", "valid_time"])
            .rename({"time_and_frame": "time"})
        )
        per_em_datasets.append(em_ds)

    ds = xr.concat(per_em_datasets, dim="ensemble_member")

    return ds


def open_concat_sample_datasets(sample_runs, split, ensemble_members, samples_per_run):
    sample_datasets = []
    for sample_run in sample_runs:
        per_var_sample_datasets = [
            open_samples_ds(
                run_name=sample_src["fq_model_id"],
                checkpoint_id=sample_src["checkpoint"],
                dataset_name=sample_src["dataset"],
                input_xfm_key=sample_src["input_xfm"],
                config_hash=sample_src.get(
                    "config_hash", None
                ),  # Optional config hash for older samples
                split=split,
                ensemble_members=ensemble_members,
                num_samples=samples_per_run,
                deterministic=sample_run["deterministic"],
            )[f"{var}"]
            for sample_src in sample_run["sample_specs"]
            for var in sample_src["variables"]
        ]

        sample_datasets.append(xr.merge(per_var_sample_datasets, join="inner"))

    samples_ds = xr.concat(
        sample_datasets, pd.Index([sr["label"] for sr in sample_runs], name="model")
    )

    if "sample_id" not in samples_ds.dims:
        samples_ds = samples_ds.expand_dims("sample_id")

    return samples_ds


def attach_derived_variables(ds, conf, prefixes=["target", "pred"]):
    for var, argsconf in conf.items():

        parts = argsconf[0].split(".")
        module_name, function_name = ".".join(parts[:-1]), parts[-1]
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

        for prefix in prefixes:

            kwargs = {
                argname: ds[f"{prefix}_{val}"] for argname, val in argsconf[1].items()
            }

            ds[f"{prefix}_{var}"] = function(**kwargs)

    return ds


def tp_from_time(x):
    for tp_key, (tp_start, tp_end) in TIME_PERIODS.items():
        if (x >= tp_start) and (x <= tp_end):
            return tp_key
    raise RuntimeError(f"No time period for {x}")


def attach_eval_coords(ds):
    time_period_coord_values = xr.apply_ufunc(
        tp_from_time, ds["time"], input_core_dims=None, vectorize=True
    )
    ds = ds.assign_coords(time_period=("time", time_period_coord_values.data))

    dec_adjusted_year = ds["time.year"] + (ds["time.month"] == 12)
    ds = ds.assign_coords(dec_adjusted_year=("time", dec_adjusted_year.data))

    ds = ds.assign_coords(
        stratum=("time", ds["time_period"].str.cat(ds["time.season"], sep=" ").data)
    )

    ds = ds.assign_coords(
        tp_season_year=(
            "time",
            ds["time_period"]
            .str.cat(ds["time.season"], ds["dec_adjusted_year"], sep=" ")
            .data,
        )
    )

    return ds

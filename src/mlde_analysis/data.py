import importlib
import pandas as pd
import xarray as xr

from mlde_utils import (
    TIME_PERIODS,
    dataset_split_path,
    workdir_path,
    samples_path,
    samples_glob,
)

from . import display


def prep_eval_data(
    sample_configs,
    dataset_configs,
    derived_var_configs,
    eval_vars,
    split,
    ensemble_members,
    samples_per_run,
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
        samples_ds = open_concat_sample_datasets(
            sample_config,
            split=split,
            ensemble_members=ensemble_members,
            samples_per_run=samples_per_run,
        )
        for var, attrs in display.ATTRS.items():
            pvarname = f"pred_{var}"
            if pvarname in samples_ds.data_vars:
                samples_ds[pvarname] = samples_ds[pvarname].assign_attrs(attrs)

        dataset_ds = open_dataset_split(
            dataset_configs[source], split, ensemble_members
        )
        for var, attrs in display.ATTRS.items():
            tvarname = f"target_{var}"
            if tvarname in dataset_ds.data_vars:
                dataset_ds[tvarname] = dataset_ds[tvarname].assign_attrs(attrs)

        ds = xr.merge([samples_ds, dataset_ds], join="inner", compat="override")
        ds = attach_eval_coords(ds)

        ds = attach_derived_variables(ds, derived_var_configs)

        merged_ds[source] = ds

    return merged_ds, models


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


def si_to_mmday(da: xr.DataArray) -> xr.DataArray:
    # convert from kg m-2 s-1 (i.e. mm s-1) to mm day-1
    attrs = {
        "units": "mm/day",
        "grid_mapping": da.attrs.get("grid_mapping", "rotated_latitude_longitude"),
        "standard_name": "precipitation_flux",
        "long_name": f"Precip.",
    }
    return (da * 3600 * 24).assign_attrs(attrs)


def open_samples_ds(
    run_name,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    ensemble_members,
    num_samples,
    deterministic,
):

    per_em_datasets = []
    for ensemble_member in ensemble_members:
        samples_dir = samples_path(
            workdir=workdir_path(run_name),
            checkpoint=checkpoint_id,
            input_xfm=input_xfm_key,
            dataset=dataset_name,
            split=split,
            ensemble_member=ensemble_member,
        )
        sample_files_list = list(samples_glob(samples_dir))
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

        per_em_datasets.append(em_ds)

    ds = xr.concat(per_em_datasets, dim="ensemble_member")
    if "pred_pr" in ds.data_vars:
        ds["pred_pr"] = si_to_mmday(ds["pred_pr"])

    return ds


def open_dataset_split(dataset_name, split, ensemble_members="all"):
    if ensemble_members == "all":
        ds = xr.open_dataset(dataset_split_path(dataset_name, split))
    else:
        ds = xr.open_dataset(dataset_split_path(dataset_name, split)).sel(
            ensemble_member=ensemble_members
        )
    if "target_pr" in ds.data_vars:
        ds["target_pr"] = si_to_mmday(ds["target_pr"])

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
                split=split,
                ensemble_members=ensemble_members,
                num_samples=samples_per_run,
                deterministic=sample_run["deterministic"],
            )[f"pred_{var}"]
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

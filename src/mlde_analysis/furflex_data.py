import importlib
from mlde_utils import (
    TIME_PERIODS,
    FurflexDatasetMetadata,
    FurflexEmulatorOutputMetadata,
)
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

from . import display
from mlde_analysis import stats


WORKDIRS_PATH = Path(os.getenv("WORKDIRS_PATH"))


def open_dataset_split(dataset_name, split, ensemble_members="all"):
    ds = xr.open_dataset(
        FurflexDatasetMetadata(dataset_name).predictands_split_path(split),
        chunks={},
    )
    if ensemble_members != "all":
        ds = ds.sel(ensemble_member=ensemble_members)

    return ds


def open_dataset_predictors_split(dataset_name, split, ensemble_members="all"):
    ds = xr.open_dataset(
        FurflexDatasetMetadata(dataset_name).predictors_split_path(split),
        chunks={},
    )
    if ensemble_members != "all":
        ds = ds.sel(ensemble_member=ensemble_members)

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
    target_sim_key="CPM",
):
    # open dataset statistics
    # open sample sets statistics for each sample run and model
    # merge across samples runs and models
    order = 1
    models = {}
    for source, data_configs in sample_configs.items():
        models[source] = {}
        for run_config in data_configs:
            models[source][run_config["label"]] = (
                {"CCS": False, "source": source} | run_config | {"order": order}
            )
            order += 1

    merged_ds = {}
    stats_dts = {}
    sim_datasets = {}
    for source, dataset_config in dataset_configs.items():
        if source not in sample_configs:
            continue  # skip datasets that don't have corresponding sample configs

        ds = _prep_sim_data(
            dataset_config,
            source,
            split,
            ensemble_members,
            eval_vars,
            exclude_days,
            derived_var_configs=derived_var_configs,
        )

        sim_datasets[source] = ds

    target_sim_ds = sim_datasets[target_sim_key]

    # WARNING: HACK to put GCM data (currently only derived from mass data regridded to CPM 2.2km data coarsened 4x same as daily work) on same coords as target dataset (for hourly this is from CEDA). These have slightly different coords though should cover the same domain. This is a hack to make the coords match so that we can merge the datasets. This should be fixed in the future by regridding the GCM data to the same coords as the target dataset.
    if "GCM" in sim_datasets:
        sim_datasets["GCM"] = sim_datasets["GCM"].assign_coords(
            {
                target_sim_ds.cf["Y"].name: target_sim_ds.cf["Y"].copy(),
                target_sim_ds.cf["X"].name: target_sim_ds.cf["X"].copy(),
            }
        )

    for source, sample_config in sample_configs.items():
        sim_ds = sim_datasets[source]
        preds_ds, pred_stats_dt = _prep_sample_data(
            sample_config,
            split=split,
            ensemble_members=ensemble_members,
            samples_per_run=samples_per_run,
            eval_vars=eval_vars,
            target_sim_ds=target_sim_ds,
            sim_ds=sim_ds,
            derived_var_configs=derived_var_configs,
        )
        sim_stats_dt = stats.from_dataset(sim_ds, (0, 200), eval_vars)

        stats_dts[source] = xr.DataTree.from_dict(
            {
                "/sim": sim_stats_dt,
                "/pred": pred_stats_dt,
            }
        ).compute()

        sim_ds = sim_ds.rename({var: f"target_{var}" for var in eval_vars})
        preds_ds = preds_ds.rename({var: f"pred_{var}" for var in eval_vars})
        ds = xr.merge([preds_ds, sim_ds], join="inner", compat="override")

        assert len(sim_ds["time"]) == len(ds["time"]), (
            f"Different time length for dataset before and after merging with samples: "
            f"{len(ds['time'])} != {len(sim_ds['time'])}. "
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
        # for source, ds in sim_datasets.items():
        #     sim_datasets[source] = xr.DataTree.from_dict({"/": ds.rename({var: f"target_{var}" for var in eval_vars}), "/stats": stats.from_dataset(ds, (0, 200), eval_vars).compute()})

        merged_ds[source] = ds

    return merged_ds, models, stats_dts


def _prep_sim_data(
    dataset_config,
    source,
    split,
    ensemble_members,
    eval_vars,
    exclude_days,
    derived_var_configs,
):
    dataset_ds = open_dataset_split(dataset_config, split, ensemble_members)
    if source == "GCM":
        # WARNING: HACK needed to upsample GCM data to hourly to match the sample data. As with spatial coords should fix this at source using this v simple upsampling method.
        for var in eval_vars:
            if var == "pr":
                dataset_ds[var] = (
                    dataset_ds[var] * 3600 * 24
                )  # convert from kg/m2/s to mm/day
            if var in dataset_ds.data_vars:
                dataset_ds[var] = dataset_ds[var].expand_dims(
                    {"frame": np.arange(0, 24)}, axis=2
                )
                # accumlated variables like pr need to be divided by 24 to get hourly values
                if var == "pr":
                    dataset_ds[var] = dataset_ds[var] / 24.0

        # merge_time_and_frame_dims assumes time is set to midnight but for daily sim data it's set to noon
        dataset_ds["time"] = (
            dataset_ds["time"] - pd.to_timedelta(12, unit="h").to_pytimedelta()
        )
        dataset_ds = merge_time_and_frame_dims(dataset_ds)

    dataset_ds = _exclude_days(dataset_ds, exclude_days)

    for var, attrs in display.ATTRS.items():
        if var in dataset_ds.data_vars:
            dataset_ds[var] = dataset_ds[var].assign_attrs(attrs)

    dataset_ds = attach_eval_coords(dataset_ds)
    dataset_ds = attach_derived_variables(dataset_ds, derived_var_configs)
    return dataset_ds


def _prep_sample_data(
    sample_runs,
    split,
    ensemble_members,
    samples_per_run,
    eval_vars,
    target_sim_ds,
    sim_ds,
    derived_var_configs,
):
    sample_datasets = []
    sample_stats = []
    for sample_run in sample_runs:
        per_var_sample_datasets = [
            _prep_sample_set_ds(
                fq_run_id=sample_src["fq_model_id"],
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
                target_sim_ds=target_sim_ds,
                sim_ds=sim_ds,
                vars=list(set(eval_vars) & set(sample_src["variables"])),
                emulator_label=sample_run["label"],
            )
            for sample_src in sample_run["sample_specs"]
        ]

        sample_datasets.append(
            xr.merge([ds for (ds, _) in per_var_sample_datasets], join="inner")
        )
        sample_stats.append(
            xr.merge([stats for (_, stats) in per_var_sample_datasets], join="inner")
        )

    samples_ds = xr.concat(
        sample_datasets, dim="model", data_vars="minimal", coords="minimal"
    )
    samples_ds = attach_derived_variables(samples_ds, derived_var_configs)

    sample_stats_dt = xr.concat(
        sample_stats, dim="model", data_vars="minimal", coords="minimal"
    )

    return samples_ds, sample_stats_dt


def _prep_sample_set_ds(
    fq_run_id,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    ensemble_members,
    num_samples,
    deterministic,
    config_hash,
    target_sim_ds,
    sim_ds,
    vars,
    emulator_label,
):
    eo_meta = FurflexEmulatorOutputMetadata(fq_run_id=fq_run_id, base_dir=WORKDIRS_PATH)

    sample_set_path = eo_meta.sample_set_dirpath(
        checkpoint=checkpoint_id,
        # input_xfm=input_xfm_key,
        dataset=dataset_name,
        split=split,
        config_hash=config_hash,
    )
    # find all the sample runs in this sample set using directory layout
    # eventually this might need to be set in parameters
    sample_run_ids = list(map(lambda p: p.name, sample_set_path.glob("*")))

    assert len(sample_run_ids) > 0, f"{sample_set_path} has no sample files"

    if deterministic:
        num_samples = 1

    sample_run_ids = sample_run_ids[:num_samples]
    if len(sample_run_ids) < num_samples:
        raise RuntimeError(
            f"{sample_set_path} does not have {num_samples} sample files"
        )

    sample_set_ds = xr.concat(
        [
            _prep_sample_run_ds(
                eo_meta,
                sample_run_id,
                checkpoint_id=checkpoint_id,
                dataset_name=dataset_name,
                input_xfm_key=input_xfm_key,
                split=split,
                ensemble_members=ensemble_members,
                config_hash=config_hash,
                target_sim_ds=target_sim_ds,
                sim_ds=sim_ds,
                vars=vars,
            )
            for sample_run_id in sample_run_ids
        ],
        dim="sample_id",
        data_vars="minimal",
        coords="minimal",
    )

    sample_set_ds = sample_set_ds.expand_dims({"model": [emulator_label]})

    sample_set_stats = xr.concat(
        [
            _pred_sample_run_stats(
                eo_meta,
                sample_run_id,
                checkpoint_id=checkpoint_id,
                dataset_name=dataset_name,
                split=split,
                config_hash=config_hash,
                target_sim_ds=target_sim_ds,
            )
            for sample_run_id in sample_run_ids
        ],
        dim="sample_id",
        data_vars="minimal",
        coords="minimal",
    ).map_over_datasets(lambda ds: ds.expand_dims({"model": [emulator_label]}))

    return sample_set_ds, sample_set_stats


def _prep_sample_run_ds(
    eo_meta,
    sample_run_id,
    checkpoint_id,
    dataset_name,
    input_xfm_key,
    split,
    ensemble_members,
    config_hash,
    target_sim_ds,
    sim_ds,
    vars,
):
    sample_filepaths = [
        eo_meta.samples_path(
            checkpoint=checkpoint_id,
            # input_xfm=input_xfm_key,
            dataset=dataset_name,
            split=split,
            config_hash=config_hash,
            sample_run_id=sample_run_id,
            ensemble_member=ensemble_member,
        )
        for ensemble_member in ensemble_members
    ]

    sample_run_ds = xr.concat(
        [
            xr.open_dataset(sample_filepath, chunks={})
            for sample_filepath in sample_filepaths
        ],
        dim="ensemble_member",
    )
    sample_run_ds = attach_eval_coords(sample_run_ds)

    for var, attrs in display.ATTRS.items():
        if var in sample_run_ds.data_vars:
            sample_run_ds[var] = sample_run_ds[var].assign_attrs(
                sim_ds[var].attrs | attrs
            )

    sample_run_ds = _assign_xy_coords_to_samples(sample_run_ds, target_sim_ds)

    sample_run_ds = sample_run_ds.expand_dims("sample_id")

    return sample_run_ds


def _pred_sample_run_stats(
    eo_meta,
    sample_run_id,
    checkpoint_id,
    dataset_name,
    split,
    config_hash,
    target_sim_ds,
):
    sample_set_stats = xr.load_datatree(
        eo_meta.sample_run_eval_stats_path(
            checkpoint=checkpoint_id,
            dataset=dataset_name,
            split=split,
            config_hash=config_hash,
            sample_run_id=sample_run_id,
        )
    )

    sample_set_stats = sample_set_stats.map_over_datasets(
        _assign_xy_coords_to_samples, kwargs={"target_sim_ds": target_sim_ds}
    )

    sample_set_stats = sample_set_stats.map_over_datasets(
        lambda ds: ds.expand_dims("sample_id")
    )

    return sample_set_stats


def _assign_xy_coords_to_samples(ds, target_sim_ds):
    # TODO: do this a sampling time!

    # sampling uses grid_latitude and grid_longitude dim names but without coords
    # if there's no grid_latitude dim, then assume not a spatial dataset so skip it
    if "grid_latitude" not in ds.cf:
        return ds

    # rename spatial dimensions to match the target dataset
    ds = ds.rename(
        {
            "grid_latitude": target_sim_ds.cf["Y"].name,
            "grid_longitude": target_sim_ds.cf["X"].name,
        }
    )
    # assign the spatial coordinates to sample spatial dimensions to match the target dataset
    ds = ds.assign_coords(
        {
            target_sim_ds.cf["Y"].name: target_sim_ds.cf["Y"].copy(),
            target_sim_ds.cf["X"].name: target_sim_ds.cf["X"].copy(),
        }
    )
    return ds


def merge_time_and_frame_dims(ds):
    # merge time and frame dimensions in sample ds into single time dimension
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

    return ds


def attach_derived_variables(ds, conf):
    for var, argsconf in conf.items():

        parts = argsconf[0].split(".")
        module_name, function_name = ".".join(parts[:-1]), parts[-1]
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

        kwargs = {argname: ds[val] for argname, val in argsconf[1].items()}

        ds[var] = function(**kwargs)

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

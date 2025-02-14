import cftime
import math

# import matplotlib

import matplotlib
from mlde_utils import cp_model_rotated_pole

from . import (
    plot_map,
    sorted_em_time_by_mean,
    VAR_LABELS,
    display,
)


def em_timestamps(ds, percentiles, overrides={}):
    em_ts = {}
    mean_sorted_em_time_cache = {}

    # then get the other samples based on percentiles
    for sample_key, sample_percentile in percentiles.items():
        season = sample_percentile["season"]
        variable = sample_percentile["variable"]
        cache_key = f"{season} {variable}"
        # if example is defined in override can just grab it
        if sample_key in overrides:
            override = overrides[sample_key]
            em_ts[sample_key] = (
                override[0],
                cftime._cftime.Datetime360Day.strptime(
                    override[1], "%Y-%m-%d %H:%M:%S", calendar="360_day"
                ),
            )
            continue

        # avoiding having to re-sort time if already done it once
        if cache_key not in mean_sorted_em_time_cache:
            seasonal_ds = ds.sel(time=ds["time"]["time.season"] == season)
            mean_sorted_em_time = sorted_em_time_by_mean(
                seasonal_ds[f"target_{variable}"]
            )
            mean_sorted_em_time_cache[cache_key] = mean_sorted_em_time
        else:
            mean_sorted_em_time_cache[cache_key]

        em_ts[sample_key] = mean_sorted_em_time[
            min(
                len(mean_sorted_em_time) - 1,
                math.ceil(
                    len(mean_sorted_em_time) * (1 - sample_percentile["percentile"])
                ),
            )
        ]

    # add any overrides not already in the list
    for sample_key, override in overrides.items():
        if sample_key not in em_ts:
            em_ts[sample_key] = (
                override[0],
                cftime._cftime.Datetime360Day.strptime(
                    override[1], "%Y-%m-%d %H:%M:%S", calendar="360_day"
                ),
            )

    return em_ts


def _plot_sim_example(example_ds, axes, vars, sim_title, example_label, example_idx):
    pcms = {}
    for ivar, var in enumerate(vars):
        ax = axes[example_idx][ivar]
        sim_example_da = example_ds.isel(model=0)[f"target_{var}"]
        pcms[var] = plot_map(
            sim_example_da,
            ax,
            style=var,
            add_colorbar=False,
        )
        # label column
        if example_idx == 0:
            title = []
            if ivar == 0:
                title.append(sim_title)
            if len(vars) > 1:
                title.append(display.ATTRS[var]["long_name"])
            ax.set_title("\n".join(title), fontsize="small")

        # label row
        if ivar == 0:
            ax.text(
                -0.1,
                0.5,
                example_label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize="small",
                rotation=90,
            )
    return pcms


def plot_examples(
    ds,
    em_ts,
    vars,
    models,
    fig,
    sim_title,
    n_samples_per_example=2,
    inputs=["vorticity850"],
):
    n_vars = len(vars)

    examples = {
        desc: ds.sel(ensemble_member=ts[0]).sel(time=ts[1], method="nearest")
        for desc, ts in em_ts.items()
    }

    input_limits = {
        input_var: {
            "vmin": min(
                [example_ds[input_var].min() for example_ds in examples.values()]
            ),
            "vmax": max(
                [example_ds[input_var].max() for example_ds in examples.values()]
            ),
        }
        for input_var in inputs
    }

    det_models = [
        mlabel for mlabel, mconfig in models.items() if mconfig["deterministic"]
    ]

    stoch_models = [
        mlabel for mlabel, mconfig in models.items() if not mconfig["deterministic"]
    ]

    axes = fig.subplots(
        nrows=len(em_ts),
        ncols=n_vars * (1 + len(det_models) + len(stoch_models) * n_samples_per_example)
        + len(inputs),
        subplot_kw={"projection": cp_model_rotated_pole},
    )
    if len(axes.shape) == 1:
        axes = axes.reshape(1, -1)

    bilinear_present = any(map(lambda x: "Bilinear" in x, det_models))

    for tsi, (desc, ts) in enumerate(em_ts.items()):
        ts_ds = ds.sel(ensemble_member=ts[0]).sel(time=ts[1], method="nearest")
        print(f"{desc} sample requested for EM{ts[0]} on {ts[1]}")
        if (
            ts_ds["ensemble_member"].data.item() != ts[0]
            or ts_ds["time"].data.item() != ts[1]
        ):
            print(
                f"{desc} sample actually  for EM{ts_ds['ensemble_member'].data.item()} on {ts_ds['time'].data.item()}"
            )

        pcms = _plot_sim_example(ts_ds, axes, vars, sim_title, desc, tsi)

        for input_idx, input_var in enumerate(inputs):
            ax = axes[tsi][n_vars + bilinear_present + input_idx]
            # contours = ts_ds[input_var].plot.contour(ax=ax, add_colorbar=False, cmap="Greys")
            # ax.clabel(contours, inline=True, fontsize="small")
            # ax.set_title("")
            # ax.coastlines(**{"resolution": "10m", "linewidth": 0.3})
            input_pcm = plot_map(
                ts_ds[input_var],
                ax,
                style=None,
                # cmap="coolwarm",
                cmap=matplotlib.colormaps.get_cmap("coolwarm").resampled(11),
                norm=matplotlib.colors.CenteredNorm(
                    vcenter=0,
                    halfrange=max(
                        abs(input_limits[input_var]["vmin"]),
                        abs(input_limits[input_var]["vmax"]),
                    ),
                ),
                add_colorbar=False,
                # **input_limits[input_var],
            )
            # label column
            if tsi == 0:
                ax.set_title(f"Example\ncoarse\ninput", fontsize="small")

        for mi, model in enumerate(stoch_models):
            for sample_idx in range(n_samples_per_example):
                for ivar, var in enumerate(vars):
                    icol = (
                        n_vars
                        + bilinear_present * n_vars
                        + len(inputs)
                        + mi * n_samples_per_example * n_vars
                        + sample_idx * n_vars
                        + ivar
                    )
                    ax = axes[tsi][icol]
                    plot_map(
                        ts_ds.sel(model=model).isel(sample_id=sample_idx)[
                            f"pred_{var}"
                        ],
                        ax,
                        style=var,
                        add_colorbar=False,
                    )

                    if tsi == 0:
                        title = []
                        if ivar == 0:
                            if sample_idx == 0:
                                title.append(f"{model}")
                            title.append(f"Sample {sample_idx+1}")
                        if len(vars) > 1:
                            title.append(display.ATTRS[var]["long_name"])

                        ax.set_title("\n".join(title), fontsize="small")

        det_model_offset = 0
        for mi, model in enumerate(det_models):
            for ivar, var in enumerate(vars):
                if "Bilinear" in model:
                    icol = 1
                    det_model_offset = -1
                else:
                    icol = (
                        n_vars
                        + bilinear_present * n_vars
                        + len(inputs)
                        + len(stoch_models) * n_samples_per_example * n_vars
                        + (mi + det_model_offset) * n_vars
                        + ivar
                    )

                ax = axes[tsi][icol]
                plot_map(
                    ts_ds.sel(model=model).isel(sample_id=0)[f"pred_{var}"],
                    ax,
                    style=var,
                    add_colorbar=False,
                )
                if tsi == 0:
                    title = []
                    if ivar == 0:
                        title.append(f"{model}")
                    if len(vars) > 1:
                        title.append(display.ATTRS[var]["long_name"])
                    ax.set_title("\n".join(title), fontsize="small")

    input_cb = fig.colorbar(
        input_pcm,
        ax=axes,
        location="bottom",
        orientation="horizontal",
        shrink=0.5,
        extend="neither",
    )
    input_cb.formatter.set_powerlimits((0, 0))
    input_cb.formatter.set_useMathText(True)
    input_cb.formatter.set_useOffset(False)
    input_cb.ax.tick_params(labelsize="x-small")
    input_cb.ax.ticklabel_format(useOffset=False, useMathText=True)
    input_cb.set_label("Rel. Vort. @ 850hPa ($s^{-1}$)", fontsize="x-small")
    input_cb.ax.xaxis.get_offset_text().set_fontsize("x-small")

    for var in reversed(vars):
        cb = fig.colorbar(
            pcms[var],
            ax=axes,
            location="bottom",
            orientation="horizontal",
            shrink=0.8,
            extend="both",
        )
        cb.ax.tick_params(axis="both", which="major", labelsize="small")
        cb.set_label(VAR_LABELS[var], fontsize="small")

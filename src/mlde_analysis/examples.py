import cftime
import math
import numpy as np
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


def _plot_sim_example(
    example_ds, axes, vars, sim_title, example_label, show_title=False, style_prefix=""
):
    pcms = {}
    for ivar, var in enumerate(vars):
        ax = axes[ivar]
        sim_example_da = example_ds[f"target_{var}"]
        pcms[var] = plot_map(
            sim_example_da,
            ax,
            style=f"{style_prefix}{var}",
            add_colorbar=False,
        )
        # label column
        if show_title:
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


def _plot_input(ax, input_da, limits, show_title=False):
    input_da.coarsen(grid_latitude=8, grid_longitude=8).mean().interp_like(
        input_da, kwargs={"fill_value": "extrapolate"}
    ).plot.contour(
        ax=ax,
        add_colorbar=False,
        colors="gray",
        levels=np.linspace(-1e-4, 1e-4, 11),
        linewidths=0.5,
        negative_linestyles="dashed",
    )
    ax.set_title("")
    ax.coastlines(**{"resolution": "10m", "linewidth": 0.3})
    # input_pcm = plot_map(
    #     input_da,
    #     ax,
    #     style=None,
    #     cmap=matplotlib.colormaps.get_cmap("coolwarm").resampled(11),
    #     norm=matplotlib.colors.CenteredNorm(
    #         vcenter=0,
    #         halfrange=max(
    #             abs(limits["vmin"]),
    #             abs(limits["vmax"]),
    #         ),
    #     ),
    #     add_colorbar=False,
    #     # **limits,
    # )

    # label column
    if show_title:
        ax.set_title(f"Example\ncoarse\ninput", fontsize="small")

    return None


def _plot_example(
    axes,
    tsi,
    desc,
    ts,
    ds,
    stoch_models,
    det_models,
    vars,
    inputs,
    style_prefix,
    sim_title,
    examples_sample_idxs,
    n_samples_per_example,
    n_vars,
    bilinear_present,
    input_limits,
):
    ts_ds = ds.sel(ensemble_member=ts[0]).sel(time=ts[1], method="nearest")
    print(f"{desc} sample requested for EM{ts[0]} on {ts[1]}")
    if (
        ts_ds["ensemble_member"].data.item() != ts[0]
        or ts_ds["time"].data.item() != ts[1]
    ):
        print(
            f"{desc} sample actually  for EM{ts_ds['ensemble_member'].data.item()} on {ts_ds['time'].data.item()}"
        )

    pcms = _plot_sim_example(
        ts_ds.isel(model=0),
        axes[tsi],
        vars,
        sim_title,
        desc,
        tsi == 0,
        style_prefix=style_prefix,
    )

    for input_idx, input_var in enumerate(inputs):
        ax = axes[tsi][n_vars + bilinear_present + input_idx]
        input_pcm = _plot_input(ax, ts_ds[input_var], input_limits[input_var], tsi == 0)

    for mi, model in enumerate(stoch_models):
        for si, sample_idx in enumerate(examples_sample_idxs):
            for ivar, var in enumerate(vars):
                icol = (
                    n_vars
                    + bilinear_present * n_vars
                    + len(inputs)
                    + mi * n_samples_per_example * n_vars
                    + si * n_vars
                    + ivar
                )
                ax = axes[tsi][icol]
                plot_map(
                    ts_ds.sel(model=model).isel(sample_id=sample_idx)[f"pred_{var}"],
                    ax,
                    style=f"{style_prefix}{var}",
                    add_colorbar=False,
                )

                if tsi == 0:
                    title = []
                    if ivar == 0:
                        if si == 0:
                            title.append(f"{model}")
                        title.append(f"Sample {si+1}")
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
                style=f"{style_prefix}{var}",
                add_colorbar=False,
            )
            if tsi == 0:
                title = []
                if ivar == 0:
                    title.append(f"{model}")
                if len(vars) > 1:
                    title.append(display.ATTRS[var]["long_name"])
                ax.set_title("\n".join(title), fontsize="small")
    return input_pcm, pcms


def plot_examples(
    ds,
    em_ts,
    vars,
    models,
    fig,
    sim_title,
    examples_sample_idxs=2,
    inputs=["vorticity850"],
    style_prefix="",
):
    n_vars = len(vars)
    if isinstance(examples_sample_idxs, list):
        n_samples_per_example = len(examples_sample_idxs)
    else:
        n_samples_per_example = examples_sample_idxs
        examples_sample_idxs = range(examples_sample_idxs)

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
        input_pcm, pcms = _plot_example(
            axes,
            tsi,
            desc,
            ts,
            ds,
            stoch_models,
            det_models,
            vars,
            inputs,
            style_prefix,
            sim_title,
            examples_sample_idxs,
            n_samples_per_example,
            n_vars,
            bilinear_present,
            input_limits,
        )

    if input_pcm is not None:
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

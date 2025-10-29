import matplotlib
from mlde_utils import cp_model_rotated_pole

from . import plot_map

VAR_LABELS = {
    "pr": "Precip (mm/day)",
    "tmean150cm": "Temp (K)",
    "relhum150cm": "Rel. Humidity (%)",
    "swbgt": "Simplified WBGT (C)",
}


def pp_plot_examples(
    ds,
    em_ts,
    vars,
    models,
    fig,
    sim_title,
    examples_sample_idxs=2,
    inputs=["vorticity850"],
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

    bilinear_present = any(map(lambda x: "Bilinear" in x, det_models))

    for tsi, (desc, ts) in enumerate(em_ts.items()):
        ts_ds = ds.sel(ensemble_member=ts[0]).sel(time=ts[1], method="nearest")
        print(f"{desc} sample requested for EM{ts[0]} on {ts[1]}")
        if (
            ts_ds["ensemble_member"].data.item() != ts[0]
            or ts_ds["time"].data.item() != ts[1]
        ):
            print(
                f"{desc} sample actually for EM{ts_ds['ensemble_member'].data.item()} on {ts_ds['time'].data.item()}"
            )

        pcms = {}

        for ivar, var in enumerate(vars):
            ax = axes[tsi][ivar]
            pcms[var] = plot_map(
                ts_ds.isel(model=0)[f"target_{var}"],
                ax,
                style=f"{var}",
                add_colorbar=False,
            )
            # label column
            if tsi == 0 and ivar == 0:
                ax.set_title(sim_title, fontsize="small")

            # label row
            if ivar == 0:
                ax.text(
                    -0.1,
                    0.5,
                    desc,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize="small",
                    rotation=90,
                )

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
                ax.set_title(f"Example coarse\ninput", fontsize="small")

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
                        ts_ds.sel(model=model).isel(sample_id=sample_idx)[
                            f"pred_{var}"
                        ],
                        ax,
                        style=f"{var}",
                        add_colorbar=False,
                    )

                    if tsi == 0 and ivar == 0:
                        ax.set_title(f"Sample {si+1}", fontsize="small")

                        if si == 0:
                            fig.text(
                                1,
                                1.35,
                                f"{model}",
                                ha="center",
                                fontsize="small",
                                transform=ax.transAxes,
                            )

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
                    style=f"{var}",
                    add_colorbar=False,
                )
                if tsi == 0:
                    ax.set_title(f"{model}", fontsize="small")

    input_cb = fig.colorbar(
        input_pcm,
        ax=axes,
        location="bottom",
        orientation="horizontal",
        shrink=0.5,
        extend="neither",
    )
    # input_cb.formatter.set_powerlimits((0, 0))
    input_cb.formatter.set_useMathText(True)
    input_cb.formatter.set_useOffset(False)
    input_cb.ax.tick_params(labelsize="x-small")
    input_cb.ax.ticklabel_format(useOffset=False, useMathText=True)
    input_cb.set_label("Rel. Vort. @ 850hPa ($s^{-1}$)", fontsize="x-small")
    input_cb.ax.xaxis.get_offset_text().set_fontsize("x-small")

    # if var == "pr":
    for var in vars:
        cb = fig.colorbar(
            pcms[var],
            ax=axes,
            location="bottom",
            orientation="horizontal",
            shrink=0.8,
            extend="both",
        )
        # cb.set_label("Relative mean change\n(percent)", fontsize="small")
        cb.ax.tick_params(axis="both", which="major", labelsize="small")
        # ax = fig.add_axes([0.05, -0.05, 0.95, 0.05])
        # cb = matplotlib.colorbar.Colorbar(
        #     ax, cmap=precip_cmap, norm=precip_norm, orientation="horizontal"
        # )
        # cb.ax.set_xticks(precip_clevs)
        # cb.ax.set_xticklabels(precip_clevs, fontsize="small")
        # cb.ax.tick_params(axis="both", which="major")
        cb.set_label(VAR_LABELS[var], fontsize="small")

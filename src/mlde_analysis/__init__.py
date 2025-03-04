import string
import matplotlib
import matplotlib.pyplot as plt
import metpy.plots.ctables
import numpy as np
import seaborn as sns
import xarray as xr

from mlde_utils import cp_model_rotated_pole

# precip_clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
#      50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
# precip_norm, precip_cmap = metpy.plots.ctables.registry.get_with_boundaries('precipitation', precip_clevs)
rainbow_precip_clevs = [
    0,
    0.1,
    1,
    2.5,
    5,
    7.5,
    10,
    15,
    20,
    30,
    40,
    50,
    70,
    100,
    150,
    200,
]
rainbow_precip_cmap = matplotlib.colors.ListedColormap(
    metpy.plots.ctables.colortables["precipitation"][: len(rainbow_precip_clevs) - 1],
    "precipitation",
)
rainbow_precip_norm = matplotlib.colors.BoundaryNorm(
    rainbow_precip_clevs, rainbow_precip_cmap.N
)
accessible_precip_clevs = [
    0.1,
    1,
    2.5,
    5,
    7.5,
    10,
    15,
    20,
    30,
    40,
    50,
    70,
    100,
    150,
    200,
]
accessible_precip_cmap = (
    matplotlib.colormaps.get_cmap("YlGnBu")
    .resampled(len(accessible_precip_clevs) - 1)
    .with_extremes(under="white")
)

gist_earth_precip_cmap = (
    matplotlib.colormaps.get_cmap("gist_earth")
    .reversed()
    .resampled(len(accessible_precip_clevs) - 1)
    .with_extremes(under="white")
)

import cmweather  # noqa


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


chase_precip_cmap = (
    truncate_colormap(matplotlib.colormaps.get_cmap("ChaseSpectral"), 0.0, 0.5)
    .reversed()
    .resampled(len(accessible_precip_clevs) - 1)
    .with_extremes(under="white")
)


def precip_norm(precip_levs, precip_cmap):
    return matplotlib.colors.BoundaryNorm(precip_levs, precip_cmap.N)


tas_levs = list(
    map(
        lambda x: x + 273,
        [
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            45,
            50,
        ],
    )
)
tas_cmap = matplotlib.colormaps.get_cmap("coolwarm")
tas_cmap = matplotlib.colormaps.get_cmap("turbo").resampled(len(tas_levs) - 1)
tas_norm = matplotlib.colors.BoundaryNorm(tas_levs, tas_cmap.N, clip=True)
tas_norm = matplotlib.colors.TwoSlopeNorm(283, vmin=263, vmax=313)
tas_norm = matplotlib.colors.Normalize(vmin=263, vmax=313)

pr_bias_cmap = matplotlib.colormaps.get_cmap("BrBG")
pr_bias_levels = matplotlib.ticker.MaxNLocator(nbins=19).tick_values(-40.0, 40.0)
pr_bias_norm = matplotlib.colors.BoundaryNorm(
    pr_bias_levels, ncolors=pr_bias_cmap.N, clip=True
)

bias_error_cmap = matplotlib.colormaps.get_cmap("RdBu")
bias_error_levels = matplotlib.ticker.MaxNLocator(nbins=19).tick_values(-40.0, 40.0)
bias_error_norm = matplotlib.colors.BoundaryNorm(
    bias_error_levels, ncolors=bias_error_cmap.N, clip=True
)

tas_bias_cmap = matplotlib.colormaps.get_cmap("coolwarm")
tas_bias_levels = matplotlib.ticker.MaxNLocator(nbins=19).tick_values(-1.0, 1.0)
tas_bias_norm = matplotlib.colors.BoundaryNorm(
    tas_bias_levels, ncolors=tas_bias_cmap.N, extend=True
)

hurs_bias_cmap = matplotlib.colormaps.get_cmap("coolwarm")
hurs_bias_levels = matplotlib.ticker.MaxNLocator(nbins=19).tick_values(-5.0, 5.0)
hurs_bias_norm = matplotlib.colors.BoundaryNorm(
    hurs_bias_levels, ncolors=hurs_bias_cmap.N, extend=True
)

swgbt_bias_cmap = matplotlib.colormaps.get_cmap("coolwarm")
swgbt_bias_levels = matplotlib.ticker.MaxNLocator(nbins=19).tick_values(-1.0, 1.0)
swgbt_bias_norm = matplotlib.colors.BoundaryNorm(
    swgbt_bias_levels, ncolors=swgbt_bias_cmap.N, extend=True
)

hurs_cmap = matplotlib.colormaps.get_cmap("Blues").resampled(20)
swbgt_cmap = matplotlib.colormaps.get_cmap("coolwarm").resampled(25)

STYLES = {
    "pr": {
        "cmap": accessible_precip_cmap,
        "norm": precip_norm(accessible_precip_clevs, accessible_precip_cmap),
    },
    "accessible_pr": {
        "cmap": accessible_precip_cmap,
        "norm": precip_norm(accessible_precip_clevs, accessible_precip_cmap),
    },
    "chase_pr": {
        "cmap": chase_precip_cmap,
        "norm": precip_norm(accessible_precip_clevs, chase_precip_cmap),
    },
    "gist_earth_pr": {
        "cmap": gist_earth_precip_cmap,
        "norm": precip_norm(accessible_precip_clevs, gist_earth_precip_cmap),
    },
    "rainbow_pr": {
        "cmap": rainbow_precip_cmap,
        "norm": precip_norm(rainbow_precip_clevs, rainbow_precip_cmap),
    },
    "relhum150cm": {"cmap": hurs_cmap, "vmin": 0, "vmax": 100},
    "tmean150cm": {"cmap": tas_cmap, "norm": tas_norm},
    "swbgt": {"cmap": swbgt_cmap, "vmin": -10, "vmax": 40},
    "temp": {"cmap": "coolwarm"},
    "vorticity": {"cmap": "RdBu"},
    "logBlues": {"cmap": "Blues", "norm": matplotlib.colors.LogNorm()},
    "prBias": {"cmap": pr_bias_cmap, "norm": pr_bias_norm},
    "tmean150cmBias": {"cmap": tas_bias_cmap, "norm": tas_bias_norm},
    "relhum150cmBias": {"cmap": hurs_bias_cmap, "norm": hurs_bias_norm},
    "swbgtBias": {"cmap": swgbt_bias_cmap, "norm": swgbt_bias_norm},
    "biasError": {"cmap": bias_error_cmap, "norm": bias_error_norm},
}

VAR_LABELS = {
    "pr": "Precip. (mm/day)",
    "tmean150cm": "Temp. (K)",
    "relhum150cm": "Rel. Humidity (%)",
    "swbgt": "Simplified WBGT (C)",
}

SUBREGIONS = {
    "SE": dict(grid_latitude=slice(10, 26), grid_longitude=slice(38, 54)),
    "NW": dict(grid_latitude=slice(44, 60), grid_longitude=slice(18, 34)),
}


def create_map_fig(grid_spec, width=None, height=None):
    if width is None:
        width = len(grid_spec[0]) * 3.5
    if height is None:
        height = len(grid_spec) * 3.5
    subplot_kw = dict(projection=cp_model_rotated_pole)
    return plt.subplot_mosaic(
        grid_spec,
        figsize=(width, height),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )


def plot_map(
    da, ax, title="", style="logBlues", add_colorbar=False, cl_kwargs=None, **kwargs
):
    if style is not None:
        kwargs = STYLES[style] | kwargs
    pcm = da.plot.pcolormesh(ax=ax, add_colorbar=add_colorbar, **kwargs)
    ax.set_title(title)
    ax.coastlines(**({"resolution": "10m", "linewidth": 0.3} | (cl_kwargs or {})))
    # ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, x_inline=False, y_inline=False)#, xlabel_style=dict(fontsize=24), ylabel_style=dict(fontsize=24))
    return pcm


def freq_density_plot(
    ax,
    pred_pr,
    target_pr,
    title="Log density of samples and CPM precip",
    target_label="CPM",
    grouping_key="model",
    diagnostics=False,
):
    hrange = (
        min(pred_pr.min().values, target_pr.min().values),
        max(pred_pr.max().values, target_pr.max().values),
    )
    _, bins, _ = target_pr.plot.hist(
        ax=ax,
        bins=50,
        density=True,
        color="black",
        alpha=0.2,
        label=target_label,
        log=True,
        range=hrange,
    )
    for group, group_pr in pred_pr.groupby(grouping_key):
        group_pr.plot.hist(
            ax=ax,
            bins=bins,
            density=True,
            alpha=0.75,
            histtype="step",
            label=f"{group}",
            log=True,
            range=hrange,
            linewidth=2,
            linestyle="-",
        )

    ax.set_title(title)
    ax.set_xlabel("Precip (mm day-1)")
    ax.tick_params(axis="both", which="major")
    if diagnostics:
        text = f"""
        # Timestamps: {pred_pr["time"].count().values}
        # Samples: {pred_pr.count().values}
        # Targets: {target_pr.count().values}
        % Samples == 0: {(((pred_pr == 0).sum()/pred_pr.count()).values*100).round()}
        % Targets == 0: {(((target_pr == 0).sum()/target_pr.count()).values*100).round()}
        % Samples < 1e-5: {(((pred_pr < 1e-5).sum()/pred_pr.count()).values*100).round()}
        % Targets < 1e-5: {(((target_pr < 1e-5).sum()/target_pr.count()).values*100).round()}
        % Samples < 0.1: {(((pred_pr < 0.1).sum()/pred_pr.count()).values*100).round()}
        % Targets < 0.1: {(((target_pr < 0.1).sum()/target_pr.count()).values*100).round()}
        % Samples < 1: {(((pred_pr < 1).sum()/pred_pr.count()).values*100).round()}
        % Targets < 1: {(((target_pr < 1).sum()/target_pr.count()).values*100).round()}
        Sample max: {pred_pr.max().values.round()}
        Target max: {target_pr.max().values.round()}
        """
        ax.text(0.7, 0.5, text, fontsize=8, transform=ax.transAxes)
    ax.legend()
    # ax.set_aspect(aspect=1)


def reasonable_quantiles(da):
    limit = int(np.log10(1 / da.size))
    print(limit)
    return np.concatenate(
        [
            np.linspace((1 - 10 ** (i + 1)) + (10**i), (1 - 10**i), 9)
            for i in range(-1, limit - 1, -1)
        ]
        + [[1.0]]
    )


def qq_plot(
    ax,
    target_quantiles,
    sample_quantiles,
    grouping_key="model",
    title="Sample vs CPM quantiles",
    xlabel="CPM precip (mm/day)",
    ylabel="Sample precip (mm/day)",
    tr=200,
    bl=0,
    guide_label="Ideal",
    show_legend=True,
    **lineplot_args,
):

    for label, group_quantiles in sample_quantiles.groupby(grouping_key, squeeze=False):
        # ax.scatter(
        #     target_quantiles,
        #     group_quantiles.mean(dim="sample_id"),
        #     **(dict(label=label, alpha=0.75, marker="x") | scatter_args),
        # )
        data = (
            group_quantiles.squeeze(grouping_key)
            .to_dataframe()
            .dropna()  # bit of a hack while have some models just for GCM and others just for CPM
            .reset_index()
        )
        # if grouping_key != "sample_id":
        #     data = data.melt(
        #         id_vars="quantile", value_vars=list(group_quantiles["sample_id"].values)
        #     )
        # else:
        #     data = data.melt(id_vars="quantile", value_vars=[0])
        data = data.merge(target_quantiles.to_dataframe().reset_index())

        kwargs = (
            dict(
                errorbar=None,
                marker="X",
                alpha=0.75,
            )
            | lineplot_args
        )
        sns.lineplot(
            data=data,
            x="target_q",
            y="pred_q",
            ax=ax,
            label=label,
            **kwargs,
        )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(
        [0, 1],
        [0, 1],
        transform=ax.transAxes,
        # linewidth=1,
        color="black",
        linestyle="--",
        label=guide_label,
        alpha=0.2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    legend = ax.legend()
    if not show_legend:
        legend.remove()
    ax.set_aspect(aspect=1)


def sorted_em_time_by_mean(da):
    em_time_da = da.stack(em_time=["ensemble_member", "time"])
    # std = em_time_ds["target_pr"].groupby("em_time").std(...)
    # std = seasonal_ds["target_pr"].std(dim=["grid_longitude", "grid_latitude"])#/merged_ds.sel(source="CPM")["target_pr"].mean(dim=["grid_longitude", "grid_latitude"])
    # std_sorted_em_time = std.sortby(-std)["em_time"].values
    mean = em_time_da.groupby("em_time").mean(...)
    return mean.sortby(-mean)["em_time"].values


def distribution_figure(
    ds,
    target_pr,
    quantiles,
    quantile_dims,
    grouping_key="model",
    density_kwargs=dict(),
    qq_kwargs=dict(),
):
    fig, axes = plt.subplot_mosaic(
        [["Density"]], figsize=(7, 3.5), constrained_layout=True
    )

    ax = axes["Density"]

    freq_density_plot(
        ax, ds["pred_pr"], target_pr, grouping_key=grouping_key, **density_kwargs
    )
    plt.show()

    fig, axes = plt.subplot_mosaic(
        [["Quantiles"]], figsize=(3.5, 3.5), constrained_layout=True
    )

    ax = axes["Quantiles"]

    cpm_quantiles = target_pr.quantile(quantiles, dim=quantile_dims)

    sample_quantiles = ds["pred_pr"].quantile(quantiles, dim=quantile_dims)
    qq_plot(
        ax,
        cpm_quantiles,
        sample_quantiles,
        grouping_key=grouping_key,
        **({"title": None} | qq_kwargs),
    )
    plt.show()

    fig, axes = plt.subplot_mosaic(
        [ds["model"].values], figsize=(10.5, 3.5), constrained_layout=True
    )
    for model, model_quantiles in sample_quantiles.groupby("model", squeeze=False):
        qq_plot(
            axes[model],
            cpm_quantiles,
            model_quantiles.squeeze("model"),
            title=model,
            grouping_key="sample_id",
            alpha=0.5,
            show_legend=False,
        )
    plt.show()


def seasonal_distribution_figure(
    samples_ds, cpm_pr_da, quantiles, quantile_dims, grouping_key="model"
):
    fig, axes = plt.subplot_mosaic(
        [["Quantiles DJF", "Quantiles MAM", "Quantiles JJA", "Quantiles SON"]],
        figsize=(14, 3.5),
        constrained_layout=True,
    )
    for season, seasonal_samples_ds in samples_ds.groupby("time.season"):
        ax = axes[f"Quantiles {season}"]
        seasonal_cpm_pr_da = cpm_pr_da.sel(time=(cpm_pr_da["time.season"] == season))

        seasonal_cpm_quantiles = seasonal_cpm_pr_da.quantile(
            quantiles, dim=quantile_dims
        )
        seasonal_sample_quantiles = seasonal_samples_ds["pred_pr"].quantile(
            quantiles, dim=quantile_dims
        )

        qq_plot(
            ax,
            seasonal_cpm_quantiles,
            seasonal_sample_quantiles,
            title=f"Sample vs CPM {season} quantiles",
            grouping_key=grouping_key,
        )
    plt.show()


def compute_gridspec(models, target_name):
    nmodels = len(models)
    ncols = 6
    nrows = nmodels // ncols
    if nmodels % ncols != 0:
        nrows = nrows + 1
    if nrows == 1:
        ncols = nmodels
    bias_gridspec = np.pad(
        models,
        (0, ncols * nrows - nmodels),
        mode="constant",
        constant_values=".",
    ).reshape(-1, ncols)
    return np.stack([np.concatenate([[target_name], row]) for row in bias_gridspec])
    # return bias_gridspec


def plot_mean_bias(ds, target_pr):
    target_mean = target_pr.mean(dim=["ensemble_member", "time"])
    sample_mean = ds["pred_pr"].mean(dim=["ensemble_member", "sample_id", "time"])
    bias = sample_mean - target_mean
    bias_ratio = 100 * bias / target_mean

    target_name = "$\\mu_{CPM}$"
    grid_spec = compute_gridspec(bias_ratio["model"].values, target_name)
    fig, axd = plt.subplot_mosaic(
        grid_spec,
        figsize=(grid_spec.shape[1] * 3.5, grid_spec.shape[0] * 3.5),
        subplot_kw=dict(projection=cp_model_rotated_pole),
        constrained_layout=True,
    )

    ax = axd[target_name]
    pcm = plot_map(
        target_mean,
        ax,
        title=target_name,
        norm=None,
        add_colorbar=False,
    )
    fig.colorbar(pcm, ax=[ax], location="right", shrink=0.8, extend="both")
    for model in bias_ratio["model"].values:
        ax = axd[model]
        pcm = plot_map(
            bias_ratio.sel(model=model),
            ax,
            title=f"{model}",
            style="prBias",
            add_colorbar=False,
        )
        global_mean_bias_ratio = (
            ds["pred_pr"].sel(model=model).mean() - target_pr.mean()
        ) / target_pr.mean()
        bias_ratio_mae = np.abs(bias_ratio.sel(model=model)).mean()
        ax.text(
            0,
            -0.05,
            "\n".join(
                [
                    f"global bias={global_mean_bias_ratio.values:.1%}",
                    f"bias mae={bias_ratio_mae.values:.1f}%",
                ]
            ),
            transform=ax.transAxes,
        )

    axes = [axd[model] for model in bias_ratio["model"].values]
    fig.colorbar(pcm, ax=axes, location="right", shrink=0.8, extend="both")

    plt.show()


def plot_std_bias(ds, target_pr):
    target_std = target_pr.std(dim=["ensemble_member", "time"])
    sample_std = ds["pred_pr"].std(dim=["ensemble_member", "sample_id", "time"])
    std_ratio = 100 * sample_std / target_std

    target_name = "$\\sigma_{CPM}$"
    grid_spec = compute_gridspec(std_ratio["model"].values, target_name)
    fig, axd = plt.subplot_mosaic(
        grid_spec,
        figsize=(grid_spec.shape[1] * 3.5, grid_spec.shape[0] * 3.5),
        subplot_kw=dict(projection=cp_model_rotated_pole),
        constrained_layout=True,
    )
    ax = axd[target_name]
    pcm = plot_map(
        target_std,
        ax,
        title=target_name,
        norm=None,
        add_colorbar=False,
    )
    fig.colorbar(pcm, ax=[ax], location="right", shrink=0.8, extend="both")
    for model in std_ratio["model"].values:
        ax = axd[model]
        pcm = plot_map(
            std_ratio.sel(model=model),
            ax,
            title=f"{model} std dev bias",
            style="discreteBias",
        )
    # plt.colorbar()
    axes = [axd[model] for model in std_ratio["model"].values]
    fig.colorbar(pcm, ax=axes, location="right", shrink=0.8, extend="both")
    plt.show()


def scatter_plots(pred_da, target_da, fig, line_props):
    axd = fig.subplot_mosaic([pred_da["model"].values], sharey=True)
    tr = max(pred_da.max(), target_da.max())
    bl = min(pred_da.min(), target_da.min())

    for idx, model in enumerate(pred_da["model"].values):
        pred_pr = pred_da.sel(model=model)

        ax = axd[model]
        ax.set_title(model, fontsize="medium")
        ax.scatter(
            x=target_da.broadcast_like(pred_pr),
            y=pred_pr,
            alpha=0.05,
            color=line_props[model]["color"],
        )

        ax.set_xlabel(
            f"CPM mean\n{xr.plot.utils.label_from_attrs(da=pred_da)}", fontsize="small"
        )
        if idx == 0:
            ax.set_ylabel(
                f"ML mean\n{xr.plot.utils.label_from_attrs(da=target_da)}",
                fontsize="small",
            )

        ax.plot(
            [bl, tr],
            [bl, tr],
            linewidth=1,
            color="black",
            linestyle="--",
            label="Ideal",
        )
        ax.set_aspect(aspect=1)
        ax.annotate(
            f"{string.ascii_lowercase[idx]}.",
            xy=(-0.05, 1.04),
            xycoords=("axes fraction", "axes fraction"),
            weight="bold",
            ha="left",
            va="bottom",
        )
    return axd

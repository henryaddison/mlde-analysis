import matplotlib.pyplot as plt
import xarray as xr

from mlde_utils import cp_model_rotated_pole

from . import plot_map

THRESHOLDS = {"pr": [0.1], "tmean150cm": [273], "relhum150cm": [50], "swbgt": [20]}


def threshold_exceeded_prop(da, threshold):
    example_dims = set(da.dims) - set(["grid_latitude", "grid_longitude"])
    return (
        100 * (da > threshold).sum(dim=example_dims) / da.count(dim=example_dims)
    ).rename("% Threshold exceeded")


def threshold_exceeded_prop_error(sample_da, target_da, threshold):
    return (
        threshold_exceeded_prop(target_da, threshold)
        - threshold_exceeded_prop(sample_da, threshold)
    ).rename("% Threshold exceeded error")


def threshold_exceeded_prop_change(da, threshold):
    from_da = da.where(da["time_period"] == "historic", drop=True)
    to_da = da.where(da["time_period"] == "future", drop=True)

    from_prop = threshold_exceeded_prop(from_da, threshold=threshold).rename(
        "% Threshold exceeded (historic)"
    )
    to_prop = threshold_exceeded_prop(to_da, threshold=threshold).rename(
        "% Threshold exceeded (future)"
    )

    change = (to_prop - from_prop).rename("change in % threshold exceeded")
    relative_change = (change / from_prop * 100).rename(
        "relative change in % threshold exceeded"
    )

    return xr.merge([from_prop, to_prop, change, relative_change])


def threshold_exceeded_prop_stats(
    sample_das,
    target_da,
    threshold,
    threshold_exceeded_prop_statistic=threshold_exceeded_prop,
):
    all_stats = []

    for season in ("Annual", "DJF", "MAM", "JJA", "SON"):
        if season == "Annual":
            season_mask = {}
        else:
            season_mask = {"time": target_da["time"]["time.season"] == season}

        seasonal_cpm_stats = threshold_exceeded_prop_statistic(
            target_da.sel(season_mask), threshold=threshold
        ).expand_dims(model=["CPM"])
        seasonal_stats = xr.concat(
            [
                threshold_exceeded_prop_statistic(
                    sample_da.sel(season_mask),
                    threshold=threshold,
                )
                for sample_da in sample_das
            ]
            + [seasonal_cpm_stats],
            dim="model",
        ).expand_dims(dim={"season": [season]})

        all_stats.append(seasonal_stats)

    return xr.concat(all_stats, dim="season")


def plot_threshold_exceedence_errors(threshold_exceedence_stats, style="raw"):
    nmodels = len(threshold_exceedence_stats["model"])
    fig = plt.figure(
        layout="constrained", figsize=((1 + nmodels) * 2.5, (1 + nmodels) * 2.5)
    )

    seasons = list(threshold_exceedence_stats["season"].data)

    spec = []
    for season in seasons:
        spec.extend(
            [
                [f"{season} CPM"]
                + [
                    f"{season} {model}"
                    for model in threshold_exceedence_stats["model"].data
                    if model != "CPM"
                ],
                [f"."]
                + [
                    f"{season} CPM - {model}"
                    for model in threshold_exceedence_stats["model"].data
                    if model != "CPM"
                ],
            ]
        )

    axd = fig.subplot_mosaic(spec, subplot_kw={"projection": cp_model_rotated_pole})

    if style == "raw":
        plot_map_kwargs = {"vmin": 20, "vmax": 80, "style": None}
    elif style == "change":
        plot_map_kwargs = {"vmin": -20, "center": 0, "style": None, "cmap": "BrBG"}

    for season in seasons:
        ax = axd[f"{season} CPM"]
        ax.text(
            -0.1,
            0,
            season,
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize="large",
            fontweight="bold",
        )

        for label, model_threxc_prop in threshold_exceedence_stats.sel(
            season=season
        ).groupby("model", squeeze=False):
            ax = axd[f"{season} {label}"]
            plot_map(
                model_threxc_prop.squeeze("model"),
                ax,
                title=f"{label}",
                add_colorbar=True,
                **plot_map_kwargs,
            )
            ax.set_title(label, fontsize="medium")

            if label != "CPM":
                label = f"CPM - {label}"
                ax = axd[f"{season} {label}"]
                plot_map(
                    (
                        threshold_exceedence_stats.sel(season=season, model="CPM")
                        - model_threxc_prop.squeeze("model")
                    ).rename("Difference"),
                    ax,
                    title=f"{label}",
                    style=None,
                    add_colorbar=True,
                    vmin=-20,
                    center=0,
                    cmap="RdBu",
                )
                ax.set_title(label, fontsize="medium")

    return fig, axd

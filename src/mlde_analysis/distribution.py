from collections import defaultdict
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy
import xarray as xr

from mlde_utils import cp_model_rotated_pole

from mlde_analysis import plot_map


def mean_bias(sample_da, cpm_da, normalize=False):
    sample_dims = set(["ensemble_member", "sample_id", "time"]) & set(sample_da.dims)

    sample_summary = sample_da.mean(dim=sample_dims)

    truth_dims = set(["ensemble_member", "sample_id", "time"]) & set(cpm_da.dims)

    cpm_summary = cpm_da.mean(dim=truth_dims)

    raw_bias = sample_summary - cpm_summary

    if normalize:
        return (
            (100 * raw_bias / cpm_summary)
            .rename("Relative bias [%]")
            .assign_attrs({"long_name": "Bias", "units": cpm_da.attrs["units"]})
        )
    else:
        return raw_bias.rename(f"Bias [{cpm_da.attrs['units']}]").assign_attrs(
            {"long_name": "Bias", "units": cpm_da.attrs["units"]}
        )


def std_bias(sample_da, cpm_da, normalize=False):
    sample_dims = set(["ensemble_member", "sample_id", "time"]) & set(sample_da.dims)
    sample_summary = sample_da.std(dim=sample_dims)

    truth_dims = set(["ensemble_member", "sample_id", "time"]) & set(cpm_da.dims)
    cpm_summary = cpm_da.std(dim=truth_dims)

    raw_bias = sample_summary - cpm_summary

    if normalize:
        return (
            (100 * raw_bias / cpm_summary)
            .rename("Relative bias [%]")
            .assign_attrs({"long_name": "Bias", "units": cpm_da.attrs["units"]})
        )
    else:
        return raw_bias.rename(f"Bias [{cpm_da.attrs['units']}]").assign_attrs(
            {"long_name": "Bias", "units": cpm_da.attrs["units"]}
        )


def rms_mean_bias(sample_da, cpm_da, normalize=False):
    return np.sqrt((mean_bias(sample_da, cpm_da, normalize=normalize) ** 2).mean())


def rms_std_bias(sample_da, cpm_da, normalize=False):
    return np.sqrt((std_bias(sample_da, cpm_da, normalize=normalize) ** 2).mean())


def normalized_mean_bias(sample_da, cpm_da):
    return mean_bias(sample_da, cpm_da, normalize=True)


def normalized_std_bias(sample_da, cpm_da):
    return std_bias(sample_da, cpm_da, normalize=True)


def xr_hist(da, bins, **kwargs):
    def _np_hist(da, bins, **kwargs):
        return np.histogram(da, bins=bins, density=True, **kwargs)[0]

    return xr.apply_ufunc(
        _np_hist,
        da,
        input_core_dims=[da.dims],  # list with one entry per arg
        output_core_dims=[["bins"]],
        vectorize=True,
        kwargs=kwargs | dict(bins=bins),
    ).rename("frequency_density")


def hist_dist(hist_da, target_hist_da):
    return xr.apply_ufunc(
        scipy.spatial.distance.jensenshannon,
        hist_da,
        target_hist_da,
        input_core_dims=[["bins"], ["bins"]],  # list with one entry per arg
        # vectorize=True,
    ).rename("JS_distance")


DIST_THRESHOLDS = defaultdict(
    list,
    {
        "pr": [0.1, 25, 75, 125],
        "relhum150cm": [35, 100],
        "tmean150cm": [273, 300],
        "swbgt": [5, 25],
    },
)


def compute_metrics(da, cpm_da, thresholds=[0.1, 25, 75, 125]):
    rms_mean_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_mean_bias, cpm_da=cpm_da, normalize=False)
        .rename(f"RMS Mean Bias ({cpm_da.attrs['units']})")
    )
    rms_std_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_std_bias, cpm_da=cpm_da, normalize=False)
        .rename(f"RMS Std Dev Bias ({cpm_da.attrs['units']})")
    )

    relative_rms_mean_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_mean_bias, cpm_da=cpm_da, normalize=True)
        .rename("Relative RMS Mean Bias (%)")
    )
    relative_rms_std_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_std_bias, cpm_da=cpm_da, normalize=True)
        .rename("Relative RMS Std Dev Bias (%)")
    )

    bins = np.histogram_bin_edges(cpm_da, bins=50)
    target_hist_da = xr_hist(cpm_da, bins=bins)

    model_hist_dist = (
        da.groupby("model", squeeze=False)
        .map(xr_hist, bins=bins)
        .groupby("model")
        .map(hist_dist, target_hist_da=target_hist_da)
        .rename("J-S distance")
    )

    thshd_exceedence_prop_da = xr.concat(
        [
            da.rename("emu_threshold_exceedence")
            .groupby("model", squeeze=False)
            .map(
                lambda group_da: (
                    group_da.where(group_da > threshold).count() / group_da.count()
                )
            )
            .expand_dims(dict(threshold=[threshold]))
            for threshold in thresholds
        ],
        dim="threshold",
    ).transpose("model", "threshold")
    cpm_thshd_exceedence_prop_da = xr.concat(
        [
            (cpm_da.where(cpm_da > threshold).count() / cpm_da.count()).expand_dims(
                dict(threshold=[threshold])
            )
            for threshold in thresholds
        ],
        dim="threshold",
    ).rename("cpm_threshold_exceedence")

    thshd_exceedence_ds = xr.merge(
        [
            cpm_thshd_exceedence_prop_da,
            thshd_exceedence_prop_da,
            (thshd_exceedence_prop_da - cpm_thshd_exceedence_prop_da).rename(
                "threhold_exceedence_diff"
            ),
        ]
    )

    metrics_ds = xr.merge(
        [
            rms_mean_biases.round(2),
            rms_std_biases.round(2),
            relative_rms_mean_biases.round(2),
            relative_rms_std_biases.round(2),
            model_hist_dist.round(4),
        ]
    )

    return xr.merge([metrics_ds, thshd_exceedence_ds])


def plot_freq_density(
    hist_data,
    ax,
    target_da=None,
    target_label="CPM",
    title="",
    legend=True,
    linestyle="-",
    alpha=0.95,
    linewidth=2,
    hrange=None,
    xlabel=None,
    yscale="log",
    **kwargs,
):

    if xlabel is None:
        if target_da is not None:
            xlabel = xr.plot.utils.label_from_attrs(da=target_da)
        else:
            xlabel = xr.plot.utils.label_from_attrs(da=hist_data[0]["data"])
    # xlabel = "Precip (mm/day)"

    if hrange is None:
        hrange = (
            min([d["data"].min().values for d in hist_data]),
            max([d["data"].max().values for d in hist_data]),
        )
        if target_da is not None:
            hrange = (
                min(hrange[0], target_da.min().values),
                max(hrange[1], target_da.max().values),
            )

    if target_da is not None:
        print(f"Target max: {target_da.max().values}")
    for d in hist_data:
        print(f"{d['label']} max: {d['data'].max().values}")

    bins = np.histogram_bin_edges([], bins=50, range=hrange)

    if target_da is not None:
        if yscale == "log":
            min_density = 1 / np.product(target_da.shape)
            print(min_density)
            ymin = 10 ** (math.floor(math.log10(min_density))) / 2
            print(ymin)
        elif yscale == "linear":
            ymin = 0
        else:
            ymin = None
        target_counts, bins = np.histogram(
            target_da, bins=bins, range=hrange, density=True
        )

        target_counts = xr_hist(target_da, bins, range=hrange)
        ax.stairs(
            target_counts,
            bins,
            fill=True,
            color="black",
            alpha=0.2,
            label=target_label,
        )
    else:
        ymin = None

    for pred in hist_data:
        counts = xr_hist(pred["data"], bins, range=hrange)
        ax.stairs(
            counts,
            bins,
            fill=False,
            color=pred["color"],
            alpha=pred.get("alpha", alpha),
            linestyle=pred.get("linestyle", linestyle),
            linewidth=linewidth,
            label=f"{pred['label']}",
            **kwargs,
        )

    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Freq. density")
    ax.set_ylim(ymin, None)
    ax.tick_params(axis="both", which="major")
    if legend:
        ax.legend(fontsize="small")
    ax.set_title(title)


def plot_mean_biases(mean_biases, axd, colorbar=False, **plot_map_kwargs):
    meanb_axes = []
    for i, bias in enumerate(mean_biases):
        label = bias["label"]
        bias_da = bias["data"]
        ax = axd[f"meanb {label}"]
        meanb_axes.append(ax)
        pcm = plot_map(
            bias_da,
            ax,
            title=f"{label}",
            add_colorbar=False,
            **(dict(style="prBias") | plot_map_kwargs),
        )
        ax.set_title(label, fontsize="medium")

    if colorbar:
        cb = plt.colorbar(
            pcm,
            ax=meanb_axes,
            location="bottom",
            shrink=0.8,
            extend="both",
            aspect=40,
        )
        cb.set_label(bias_da.name)
    return meanb_axes


def plot_std_biases(std_biases, axd, colorbar=True, **plot_map_kwargs):
    stddevb_axes = []
    # meanb_axes = []
    for i, bias in enumerate(std_biases):
        label = bias["label"]
        bias_da = bias["data"]
        ax = axd[f"stddevb {label}"]
        stddevb_axes.append(ax)
        # meanb_axes.append(axd[f"meanb {label}"])
        pcm = plot_map(
            bias_da,
            ax,
            title=f"{label}",
            add_colorbar=False,
            **(dict(style="prBias") | plot_map_kwargs),
        )
        ax.set_title(label, fontsize="medium")
    if colorbar:
        cb = plt.colorbar(
            pcm,
            ax=stddevb_axes,
            location="bottom",
            shrink=0.8,
            extend="both",
            aspect=40,
        )
        cb.set_label(bias_da.name)

    return stddevb_axes


def plot_distribution_figure(
    fig,
    hist_das,
    cpm_da,
    mean_bias_das,
    std_bias_das,
    modellabel2spec,
    error_ax=None,
    hrange=None,
    fd_kwargs={},
    bias_kwargs={},
):
    # re-organize data for visualizing
    hist_data = sorted(
        map(
            lambda modelgp: dict(
                data=modelgp[1].squeeze("model"),
                label=modelgp[0],
                color=modellabel2spec[modelgp[0]]["color"],
            ),
            hist_das.groupby("model", squeeze=False),
        ),
        key=lambda x: modellabel2spec[x["label"]]["order"],
    )
    mean_biases = sorted(
        map(
            lambda modelgp: dict(data=modelgp[1].squeeze("model"), label=modelgp[0]),
            mean_bias_das.groupby("model", squeeze=False),
        ),
        key=lambda x: modellabel2spec[x["label"]]["order"],
    )
    std_biases = sorted(
        map(
            lambda modelgp: dict(data=modelgp[1].squeeze("model"), label=modelgp[0]),
            std_bias_das.groupby("model", squeeze=False),
        ),
        key=lambda x: modellabel2spec[x["label"]]["order"],
    )

    meanb_axes_keys = [f"meanb {x['label']}" for x in mean_biases]
    meanb_spec = np.array(meanb_axes_keys).reshape(1, -1)

    stddevb_axes_keys = [f"stddevb {x['label']}" for x in std_biases]
    stddevb_spec = np.array(stddevb_axes_keys).reshape(1, -1)

    dist_spec = np.array(["Density"] * meanb_spec.shape[1]).reshape(1, -1)

    spec = np.concatenate([dist_spec, meanb_spec, stddevb_spec], axis=0)
    print(spec)
    axd = fig.subplot_mosaic(
        spec,
        gridspec_kw=dict(height_ratios=[3, 2, 2]),
        per_subplot_kw={
            ak: {"projection": cp_model_rotated_pole}
            for ak in meanb_axes_keys + stddevb_axes_keys
        },
    )

    ax = axd["Density"]
    plot_freq_density(
        hist_data, ax=ax, target_da=cpm_da, linewidth=1, hrange=hrange, **fd_kwargs
    )
    ax.annotate(
        "a.",
        xy=(0.04, 1.0),
        xycoords=("figure fraction", "axes fraction"),
        weight="bold",
        ha="left",
        va="bottom",
    )

    axes = plot_mean_biases(mean_biases, axd, **bias_kwargs)
    axes[0].annotate(
        "b.",
        xy=(0.04, 1.0),
        xycoords=("figure fraction", "axes fraction"),
        weight="bold",
        ha="left",
        va="bottom",
    )

    axes = plot_std_biases(std_biases, axd, **bias_kwargs)
    axes[0].annotate(
        "c.",
        xy=(0.04, 1.0),
        xycoords=("figure fraction", "axes fraction"),
        weight="bold",
        ha="left",
        va="bottom",
    )

    if error_ax is not None:

        if hrange is None:
            hrange = (
                min(
                    [d["data"].min().values for d in hist_data] + [cpm_da.min().values]
                ),
                max(
                    [d["data"].max().values for d in hist_data] + [cpm_da.max().values]
                ),
            )
        bins = np.histogram_bin_edges([], bins=50, range=hrange)
        true_counts, bins = np.histogram(cpm_da, bins=bins, range=hrange, density=True)
        mindensity = 1 / (np.product(cpm_da.shape))
        print(mindensity)
        ymin = 10 ** (math.floor(math.log10(mindensity))) / 2
        print(ymin)
        error_ax.set_ylim(ymin, None)
        error_ax.set_yscale("log")

        for pred in hist_data:
            pred_counts, bins = np.histogram(
                pred["data"], bins=bins, range=hrange, density=True
            )
            error_ax.stairs(
                np.abs(true_counts - pred_counts),
                bins,
                baseline=None,
                fill=False,
                color=pred["color"],
                alpha=pred.get("alpha", 0.95),
                linestyle=pred.get("linestyle", "-"),
                linewidth=1,
                label=f"{pred['label']}",
            )
        error_ax.legend(fontsize="small")
        error_ax.set_title("Absolute Error in freq density")
        error_ax.set_xlabel(xr.plot.utils.label_from_attrs(da=cpm_da))

    return axd

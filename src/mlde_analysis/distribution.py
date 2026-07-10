from collections import defaultdict
import functools
import math
import dask
from matplotlib import pyplot as plt
import numpy as np
import scipy
import string
import xarray as xr

from mlde_utils import cp_model_rotated_pole

from mlde_analysis import plot_map


QUANTILES = 1 - np.power(10.0, np.arange(-2, -10, -1))
PER_GRIDBOX_QUANTILES = 1 - np.power(10.0, np.arange(-2, -4, -1))
DIST_THRESHOLDS = defaultdict(
    list,
    {
        "pr": [0.1, 25, 75, 125],
        "relhum150cm": [35, 100],
        "tmean150cm": [273, 300],
        "tasmax": [273, 303],
        "swbgt": [5, 25],
    },
)


def mean_bias(sample_da, target_da, normalize=False):
    return stat_bias(sample_da, target_da, xr.DataArray.mean, normalize=normalize)


def std_bias(sample_da, target_da, normalize=False):
    return stat_bias(sample_da, target_da, xr.DataArray.std, normalize=normalize)


def stat_bias(
    sample_da,
    target_da,
    stat_func,
    normalize=False,
):
    sample_dims = set(["ensemble_member", "sample_id", "time"]) & set(sample_da.dims)

    sample_summary = stat_func(sample_da, dim=sample_dims)

    truth_dims = set(["ensemble_member", "sample_id", "time"]) & set(target_da.dims)

    cpm_summary = stat_func(target_da, dim=truth_dims)

    raw_bias = sample_summary - cpm_summary

    if normalize:
        return (
            (100 * raw_bias / cpm_summary)
            .rename("Relative bias [%]")
            .assign_attrs({"long_name": "Bias", "units": "%"})
        )
    else:
        return raw_bias.rename(f"Bias [{target_da.attrs['units']}]").assign_attrs(
            {"long_name": "Bias", "units": target_da.attrs["units"]}
        )


def rms(da: xr.DataArray) -> xr.DataArray:
    """
    Compute the root mean square of a DataArray.
    """
    return dask.array.sqrt((da**2).mean())


def rms_mean_bias(sample_da, target_da, normalize=False):
    return rms(mean_bias(sample_da, target_da, normalize=normalize))


def rms_std_bias(sample_da, target_da, normalize=False):
    return rms(std_bias(sample_da, target_da, normalize=normalize))


def rms_stat_bias(stat_bias_func, sample_da, target_da, normalize=False):
    return rms(stat_bias_func(sample_da, target_da, normalize=normalize))


def rms_q999_bias(sample_da, target_da, normalize=False):
    f_q999_bias = functools.partial(
        stat_bias, stat_func=functools.partial(xr.DataArray.quantile, q=0.999)
    )
    return rms_stat_bias(f_q999_bias, sample_da, target_da, normalize=normalize)


def normalized_mean_bias(sample_da, target_da):
    return mean_bias(sample_da, target_da, normalize=True)


def normalized_std_bias(sample_da, target_da):
    return std_bias(sample_da, target_da, normalize=True)


def xr_hist(da, bins, **kwargs):
    def _np_hist(da, bins, **kwargs):
        hist_output = np.histogram(da, bins=bins, **kwargs)
        return hist_output[0], hist_output[1]

    def _dask_hist(da, bins, **kwargs):
        hist_output = dask.array.histogram(da, bins=bins, **kwargs)
        return hist_output[0], hist_output[1]

    if isinstance(da.data, dask.array.Array):
        hist_func = _dask_hist
        extra_kwargs = {"dask": "allowed"}
    else:
        hist_func = _np_hist
        extra_kwargs = {}

    hist, bin_edges = xr.apply_ufunc(
        hist_func,  # first the function
        da,
        input_core_dims=[da.dims],  # list with one entry per arg
        output_core_dims=[["bins"], ["edge"]],
        vectorize=True,
        kwargs=dict(bins=bins, density=True) | kwargs,
        **extra_kwargs,
    )
    hist = hist.rename("frequency_density")
    return hist, bin_edges.values


def hist_dist(hist_da, target_hist_da):
    extra_args = {}
    if isinstance(hist_da.data, dask.array.Array):
        extra_args["dask"] = "allowed"
    return xr.apply_ufunc(
        scipy.spatial.distance.jensenshannon,
        hist_da.squeeze("model", drop=True),
        target_hist_da,
        input_core_dims=[["bins"], ["bins"]],  # list with one entry per arg
        # vectorize=True,
        **extra_args,
    ).rename("JS_distance")


def compute_metrics(da, target_da, thresholds=[0.1, 25, 75, 125]):
    nan_count = (
        dask.array.isnan(da)
        .groupby("model", squeeze=False)
        .sum(...)
        .rename(f"NaN Count")
    )
    target_max = target_da.max()
    max_value = (
        da.groupby("model", squeeze=False)
        .max(dim=...)
        .rename(f"Max Value ({target_da.attrs['units']})")
    )
    max_value_bias = (max_value - target_max).rename(
        f"Max Value Bias ({target_da.attrs['units']})"
    )
    target_vhi_exceedence_count = target_da.where(target_da > 60).count()
    vhi_exceedence_count = (
        da.groupby("model", squeeze=False)
        .map(lambda gda: gda.where(gda > 60).count())
        .rename(f"VHI Exceedence Count")
    )
    vhi_exceedence_bias = (vhi_exceedence_count - target_vhi_exceedence_count).rename(
        f"VHI Exceedence Count Bias"
    )
    rms_mean_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_mean_bias, target_da=target_da, normalize=False)
        .rename(f"RMS Mean Bias ({target_da.attrs['units']})")
    )
    rms_std_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_std_bias, target_da=target_da, normalize=False)
        .rename(f"RMS Std Dev Bias ({target_da.attrs['units']})")
    )

    rms_q999_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_q999_bias, target_da=target_da, normalize=False)
        .rename(f"RMS Q999 Bias ({target_da.attrs['units']})")
    )

    relative_rms_mean_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_mean_bias, target_da=target_da, normalize=True)
        .rename("Relative RMS Mean Bias (%)")
    )
    relative_rms_std_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_std_bias, target_da=target_da, normalize=True)
        .rename("Relative RMS Std Dev Bias (%)")
    )
    relative_rms_q999_biases = (
        da.groupby("model", squeeze=False)
        .map(rms_q999_bias, target_da=target_da, normalize=True)
        .rename(f"Relative RMS Q999 Bias (%)")
    )

    target_min = target_da.min().compute()
    target_max = target_da.max().compute()
    bins = np.histogram_bin_edges(
        [], bins=200, range=(target_min.item(), target_max.item())
    )
    target_hist_da, bins = xr_hist(target_da, bins=bins)
    model_hist_dist = (
        da.groupby("model", squeeze=False)
        .map(lambda x: xr_hist(x, bins=bins)[0])
        .groupby("model", squeeze=False)
        .map(hist_dist, target_hist_da=target_hist_da)
        .rename("J-S distance")
    )

    metrics_ds = xr.merge(
        [
            nan_count,
            rms_mean_biases,
            rms_std_biases,
            rms_q999_biases,
            relative_rms_mean_biases,
            relative_rms_std_biases,
            relative_rms_q999_biases,
            model_hist_dist,
            max_value,
            max_value_bias,
            vhi_exceedence_count,
            vhi_exceedence_bias,
        ],
        compat="no_conflicts",
    )

    # das = []
    # for threshold in thresholds:
    #     emu_exceedence_da = (
    #         da.groupby("model", squeeze=False)
    #         .map(
    #             lambda group_da: (
    #                 group_da.where(group_da > threshold).count() / group_da.count()
    #             )
    #         )
    #         .rename(f"Emu > {threshold}")
    #     )

    #     target_exceedence_da = (
    #         target_da.where(target_da > threshold).count() / target_da.count()
    #     ).rename(f"CPM > {threshold}")

    #     diff_da = (emu_exceedence_da - target_exceedence_da).rename(
    #         f"Emu > {threshold} - CPM > {threshold}"
    #     )
    #     das.extend([emu_exceedence_da, diff_da])

    # thshd_exceedence_ds = xr.merge(das)
    # metrics_ds = xr.merge([metrics_ds, thshd_exceedence_ds])

    return metrics_ds


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

    bins = np.histogram_bin_edges([], bins=200, range=hrange)

    if target_da is not None:
        if yscale == "log":
            min_density = 1 / np.prod(target_da.shape)
            ymin = 10 ** (math.floor(math.log10(min_density))) / 2
        elif yscale == "linear":
            ymin = 0
        else:
            ymin = None
        # target_counts, bins = np.histogram(
        #     target_da, bins=bins, range=hrange, density=True
        # )

        target_counts, bins = xr_hist(target_da, bins, range=hrange)
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
        counts, bins = xr_hist(pred["data"], bins, range=hrange)
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
    ax.set_xlabel(xlabel, fontsize="small")
    ax.set_ylabel("Freq. density", fontsize="small")
    ax.set_ylim(ymin, None)
    ax.tick_params(axis="both", which="major", labelsize="small")
    if legend:
        ax.legend(fontsize="small")
    ax.set_title(title, fontsize="small")


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


def plot_biases(biases, axes, fig, colorbar=True, **plot_map_kwargs):
    for i, bias in enumerate(biases):
        label = bias["label"]
        bias_da = bias["data"]
        ax = axes[i]
        pcm = plot_map(
            bias_da,
            ax,
            title=f"{label}",
            add_colorbar=False,
            **(dict(style="prBias") | plot_map_kwargs),
        )
        ax.set_title(label, fontsize="small")

        ax.text(
            0.99,
            0.99,
            f"{rms(bias_da).values.item():.1f}",
            fontsize="x-small",
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
                boxstyle="round,pad=0.1",
            ),
        )

    if colorbar:
        cb = fig.colorbar(
            pcm,
            ax=axes,
            location="bottom",
            pad=0.12,
            shrink=0.8,
            extend="both",
            aspect=40,
        )
        cb.set_label(bias_da.name, fontsize="small")
        cb.ax.tick_params(labelsize="small")


def plot_freq_density_figure(pred_da, target_label, modellabel2spec, fig):
    hist_data = sorted(
        map(
            lambda modelgp: dict(
                data=modelgp[1].squeeze("model"),
                label=modelgp[0],
                color=modellabel2spec[modelgp[0]]["color"],
            ),
            pred_da.groupby("model", squeeze=False),
        ),
        key=lambda x: modellabel2spec[x["label"]]["order"],
    )

    axd = fig.subplot_mosaic([["Density"]])
    ax = axd["Density"]
    plot_freq_density(
        hist_data, ax=ax, target_da=target_label, linewidth=1, yscale="log"
    )

    return ax


def plot_distribution_figure(
    fig,
    hist_das,
    target_da,
    biases_das,
    modellabel2spec,
    error_ax=None,
    hrange=None,
    height_ratio=None,
    fd_kwargs={},
    bias_kwargs={},
):
    if height_ratio is None:
        height_ratio = [3] + [1] * len(biases_das)
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
    decorated_biases = {
        bias_key: sorted(
            map(
                lambda modelgp: dict(
                    data=modelgp[1].squeeze("model"), label=modelgp[0]
                ),
                bias_das.groupby("model", squeeze=False),
            ),
            key=lambda x: modellabel2spec[x["label"]]["order"],
        )
        for bias_key, bias_das in biases_das.items()
    }

    biases_layout = {
        bias_key: [f"{bias_key} {x['label']}" for x in decbias]
        for bias_key, decbias in decorated_biases.items()
    }

    dist_spec = np.array(["Density"] * len(list(biases_layout.values())[0])).reshape(
        1, -1
    )

    spec = np.concatenate(
        [dist_spec]
        + [np.array(keys).reshape(1, -1) for keys in biases_layout.values()],
        axis=0,
    )
    axd = fig.subplot_mosaic(
        spec,
        gridspec_kw=dict(height_ratios=height_ratio),
        per_subplot_kw={
            ak: {"projection": cp_model_rotated_pole}
            for bias_keys in biases_layout.values()
            for ak in bias_keys
        },
    )

    ax = axd["Density"]
    plot_freq_density(
        hist_data, ax=ax, target_da=target_da, linewidth=1, hrange=hrange, **fd_kwargs
    )
    ax.annotate(
        "a.",
        xy=(0.04, 1.0),
        xycoords=("figure fraction", "axes fraction"),
        weight="bold",
        ha="left",
        va="bottom",
    )
    for idx, (bias_key, decbias) in enumerate(decorated_biases.items()):
        axes = [axd[f'{bias_key} {bias["label"]}'] for bias in decbias]
        show_colorbar = idx == len(decorated_biases) - 1
        plot_biases(decbias, axes, fig, colorbar=show_colorbar, **bias_kwargs)
        axes[0].annotate(
            string.ascii_lowercase[idx + 1] + ".",
            xy=(0.04, 1.0),
            xycoords=("figure fraction", "axes fraction"),
            weight="bold",
            ha="left",
            va="bottom",
        )

    if error_ax is not None:
        # TODO: make this dask-friendly (mainly by storing the computed histograms from previous steps and reusing them here, instead of recomputing them)
        if hrange is None:
            hrange = (
                min(
                    [d["data"].min().values for d in hist_data]
                    + [target_da.min().values]
                ),
                max(
                    [d["data"].max().values for d in hist_data]
                    + [target_da.max().values]
                ),
            )
        bins = np.histogram_bin_edges([], bins=200, range=hrange)
        true_counts, bins = np.histogram(
            target_da, bins=bins, range=hrange, density=True
        )
        mindensity = 1 / (np.prod(target_da.shape))
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
        error_ax.set_xlabel(xr.plot.utils.label_from_attrs(da=target_da))

    return axd

import numpy as np

from .distribution import xr_hist

# import scipy


def fc_bins():
    bin2 = np.exp(
        np.log(0.005)
        + np.sqrt(
            np.linspace(0, 99, 200)
            * ((np.square(np.log(120.0) - np.log(0.005))) / 59.0)
        )
    )
    return np.pad(bin2, (1, 0), "constant", constant_values=0) / 24.0 * 1.5


def fc_binval(bins):
    nbins = len(bins) - 1
    binval = np.zeros((nbins), dtype=float)
    for ibin in np.arange(nbins):
        binval[ibin] = (bins[ibin] + bins[ibin + 1]) / 2.0

    return binval


def compute_fractional_contribution(pr_da, bins):
    binval = fc_binval(bins)

    hist = xr_hist(pr_da, bins, density=False).assign_coords({"bins": ("bins", binval)})
    fracdist = hist * binval
    fracdist = fracdist / float(fracdist.sum()) * 100.0  # fractional distribution in %

    return fracdist


# my version
# def compute_fractional_contribution(pr_da, calchist=np.histogram):
#     hrange = (0, 250)
#     bins = np.histogram_bin_edges([], bins=50, range=hrange)
#     density, bins = calchist(pr_da, bins=bins, range=range, density=True)

#     bin_mean, _, _ = scipy.stats.binned_statistic(
#         pr_da.data.reshape(-1), pr_da.data.reshape(-1), bins=bins, range=range
#     )
#     # don't allow NaNs in bin_means - if no values for bin then frac contrib will be 0
#     bin_mean[np.isnan(bin_mean)] = 0

#     return density * bin_mean, bins


def frac_contrib_change(pr_da, bins):
    fpr = pr_da.where(pr_da["time_period"] == "future", drop=True)
    ffraccontrib = compute_fractional_contribution(fpr, bins)

    hpr = pr_da.where(pr_da["time_period"] == "historic", drop=True)
    hfraccontrib = compute_fractional_contribution(hpr, bins)

    return ffraccontrib - hfraccontrib


def plot_fractional_contribution(
    frac_contrib_data,
    ax,
    title="",
    legend=True,
    linestyle="-",
    alpha=0.95,
    linewidth=1,
    ylim=[0, 2],
    **kwargs,
):
    for pred in frac_contrib_data:
        frac_contrib = pred["data"][0]
        binval = pred["data"][1]

        ax.plot(
            binval[1:],
            frac_contrib[1:],
            color=pred["color"],
            alpha=pred.get("alpha", alpha),
            linestyle=pred.get("linestyle", linestyle),
            linewidth=linewidth,
            label=f"{pred['label']}",
            **kwargs,
        )

    ax.set_title(title)
    ax.set_xlabel("Precip (mm/day)")
    ax.set_ylabel("Fractional contrib.\n(mm/day)")
    ax.tick_params(axis="both", which="major")
    if legend:
        ax.legend(fontsize="small")
    ax.set_xscale("log")
    ax.set_xlim([0.1, 200.0])
    ax.set_ylim(ylim)  # for diff or change use ax.set_ylim([-0.4, 0.4])

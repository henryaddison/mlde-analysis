import numpy as np

# import scipy


def compute_fractional_contribution(pr_da, calchist=np.histogram):
    bin2 = np.exp(
        np.log(0.005)
        + np.sqrt(
            np.linspace(0, 99, 200)
            * ((np.square(np.log(120.0) - np.log(0.005))) / 59.0)
        )
    )
    bins = np.pad(bin2, (1, 0), "constant", constant_values=0) / 24.0 * 1.5
    NBINS = len(bins) - 1
    binval = np.zeros((NBINS), dtype=float)
    for ibin in np.arange(NBINS):
        binval[ibin] = (bins[ibin] + bins[ibin + 1]) / 2.0

    # calculating fractional distribution
    dist = np.zeros(NBINS)

    fracdist = np.zeros(NBINS)

    # for i,timeseries in enumerate(pr_da.slices('time')):
    hist = calchist(pr_da, bins)
    dist[:] = dist[:] + hist[0]

    for ibin in np.arange(NBINS):
        fracdist[ibin] = dist[ibin] * binval[ibin]
    fracdist[:] = (
        fracdist[:] / float(fracdist.sum()) * 100.0
    )  # fractional distribution in %

    return fracdist, binval


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


def plot_fractional_contribution(
    frac_contrib_data,
    ax,
    title="",
    legend=True,
    linestyle="-",
    alpha=0.95,
    linewidth=2,
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

    ax.set_xlabel("Precip (mm/day)")
    ax.set_ylabel("Fractional contrib.\n(mm/day)")
    # ax.set_ylim(ymin, None)
    ax.tick_params(axis="both", which="major")
    if legend:
        ax.legend(fontsize="small")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlim([0.1, 200.0])
    # ax.set_yscale("log")
    ax.set_ylim([0, 2])  # for diff or change use ax.set_ylim([-0.4, 0.4])


def frac_contrib_change(pr_da):
    fpr = pr_da.where(pr_da["time_period"] == "future", drop=True)
    ffraccontrib, fbinvals = compute_fractional_contribution(fpr)

    hpr = pr_da.where(pr_da["time_period"] == "historic", drop=True)
    hfraccontrib, hbinvals = compute_fractional_contribution(hpr)

    assert np.all(fbinvals == hbinvals)

    return ffraccontrib - hfraccontrib, fbinvals


def plot_fractional_contribution_change(frcontrib_change_data, ax, title):
    for pred in frcontrib_change_data:
        ax.plot(
            pred["data"][1][1:],
            pred["data"][0][1:],
            # baseline=None,
            # fill=False,
            color=pred["color"],
            alpha=pred.get("alpha", 0.95),
            linestyle=pred.get("linestyle", "-"),
            linewidth=1,
            label=f"{pred['label']}",
        )
    ax.set_title(title)
    ax.set_xlabel("Precip (mm/day)")
    ax.set_ylabel("Change in frac. contrib.")
    ax.set_ylim([-0.4, 0.4])
    ax.set_xscale("log")
    ax.set_xlim([0.1, 200.0])

    # # linthresh based on minimum value
    # linthresh = (min((map(lambda h: np.min(np.fabs(h["data"][h["data"].nonzero()])), frcontrib_change_data))))
    # print(linthresh)
    # linthresh = (10 ** math.floor(math.log10(linthresh)))/2
    # print(linthresh)

    # # linthresh based on minimum value from CPM
    # linthresh = np.min(np.fabs(frcontrib_change_data[0]["data"][frcontrib_change_data[0]["data"].nonzero()]))
    # print(linthresh)
    # linthresh = (10 ** math.floor(math.log10(linthresh)))/2
    # print(linthresh)

    # linthreshold based on single observation at reasonably high precip
    # mindensity = 1 / (np.product(CPM_DAS["pr"].shape)/3) # divide by 3 as considering single time periods
    # print(mindensity)
    # linthresh = 10 ** (math.floor(math.log10(100*mindensity))) / 2 # multiply by 100 as frac contrib is density times intensity
    # print(linthresh)

    # ax.set_yscale("symlog", linthresh=linthresh)
    ax.tick_params(axis="both", which="major")
    # ax.legend(ncols=2, fontsize="small")

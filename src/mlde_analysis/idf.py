import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize

THRESHOLD = 0.1  # threshold value in mm/hr used to mark end of a spell
INTENSITY_BINS = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
DURATIONS_BINS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    18,
    24,
    36,
    48,
    72,
    96,
    120,
]  # durations of 1 hour

# Probability based norms
# NORM = LogNorm(vmin=1e-5, vmax=0.01)
# DIFF_NORM = SymLogNorm(linthresh=0.0001, vmin=-0.1, vmax=0.1)

# Frequency based norms
NORM = LogNorm(vmin=1, vmax=1e5)
DIFF_NORM = SymLogNorm(linthresh=1, vmin=-1e4, vmax=1e4)

REL_DIFF_NORM = Normalize(vmin=-100, vmax=100)


def ucalc_distn_ndimage(data, threshold=THRESHOLD):
    id_regions, num_ids = ndimage.label(data >= threshold, structure=[1, 1, 1])

    region_intensities = ndimage.maximum(
        data, id_regions, index=np.arange(0, num_ids + 1)
    )
    # this lumps all the dry entries into one spell even though it's not contiguous
    region_durs = ndimage.labeled_comprehension(
        data,
        id_regions,
        index=np.arange(0, num_ids + 1),
        func=len,
        out_dtype=int,
        default=0,
    )

    hist = np.histogram2d(
        region_durs, region_intensities, [DURATIONS_BINS, INTENSITY_BINS]
    )
    # dry spells are lumped into one spell but should be counted as multiple spells of length 1, so we set the dry intensity, short duration bin to the "length" of the dry spells and other dry durations to 0
    hist[0][:, 0] = 0
    hist[0][0, 0] = region_durs[0]
    return hist[0]


def calc_pmf_ndimage(da):
    """
    Calculate the 2D probability mass function (PMF) of spell durations and maximum values.
    Computed for each individual time series in the input DataArray and then summed across all time series (e.g. over all grid boxes and ensemble members) to produce a single PMF for the entire DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array.

    Returns
    -------
    np.ndarray
        A 2D array representing the PMF of spell durations and maximum values.
    """
    hist = (
        xr.apply_ufunc(
            ucalc_distn_ndimage,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[["duration_bin", "intensity_bin"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={
                "output_sizes": {
                    "duration_bin": len(DURATIONS_BINS) - 1,
                    "intensity_bin": len(INTENSITY_BINS) - 1,
                }
            },
        )
        .cf.sum(dim=["ensemble_member", "X", "Y"])
        .compute()
    )
    return hist  # / hist.sum(dim=["duration_bin", "intensity_bin"]).rename(
    # "Probability Mass"
    # )


def plot_pmf(ax, pmf, title, **kwargs):
    """
    Plot the 2D probability mass function (PMF) on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the PMF.
    pmf : array-like
        The probability mass function values to plot.
    title : str
        The title for the plot.
    """
    plotted_intensity_bins = INTENSITY_BINS[1:]
    xticks = np.arange(0, len(plotted_intensity_bins), 1)
    yticks = np.arange(0, len(DURATIONS_BINS), 1)
    shw = ax.pcolormesh(
        xticks,
        yticks,
        pmf.drop_isel(intensity_bin=0),
        **kwargs,
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(plotted_intensity_bins, rotation=90, fontsize="x-small")
    ax.set_yticks(yticks)
    ax.set_yticklabels(DURATIONS_BINS, fontsize="x-small")
    ax.set_title(title)

    return shw


def plot_pmfs(
    target_pmf, pred_pmf, target_cbar=False, cbar_label="Frequency", **kwargs
):
    entries = ["target"] + list(pred_pmf["model"].values)
    cols = min(3, len(entries))
    npads = (cols - len(entries) % cols) % cols
    grid_spec = np.pad(
        np.array(entries), (0, npads), mode="constant", constant_values="."
    ).reshape(-1, cols)

    width = 2 * grid_spec.shape[1] + 1
    height = 2.5 * grid_spec.shape[0]
    if target_cbar:
        width += 0.5
    fig = plt.figure(layout="constrained", figsize=(width, height))
    axd = fig.subplot_mosaic(grid_spec, sharex=True, sharey=True)

    ax = axd["target"]
    shw = plot_pmf(ax, target_pmf, title="CPM", norm=NORM)
    if target_cbar:
        cb = fig.colorbar(
            shw,
            ax=ax,
            location="right",
            extend="max",
        )
        cb.set_label("Frequency", fontsize="small")
        cb.ax.tick_params(labelsize="x-small")

    for model, model_pmf in pred_pmf.groupby("model"):
        ax = axd[model]
        shw = plot_pmf(ax, model_pmf.squeeze(), title=model, **kwargs)
        # shw = plot_pmf(ax, (model_pmf.squeeze() - target_pmf), title=model, cmap="RdBu", norm=diff_norm)

    cb = fig.colorbar(
        shw,
        ax=[ax for k, ax in axd.items() if not (target_cbar and k == "target")],
        location="right",
        extend="max",
    )
    cb.set_label(cbar_label, fontsize="small")
    cb.ax.tick_params(labelsize="small")

    fig.supxlabel("Maximum intensity (mm/hr)")
    fig.supylabel("Spell duration (hours)")

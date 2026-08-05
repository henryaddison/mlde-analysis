import numpy as np
import xarray as xr
import scipy.ndimage as ndimage

# THRESHOLD = 0.01 # threshold value in mm/10min
# BINS = [0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0,1000.0]
THRESHOLD = 0.1  # threshold value in mm/hr used to mark end of a spell
INTENSITY_BINS = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1000.0]
AREA_BINS = [1, 2, 3, 4, 9, 16, 25, 36, 49, 64, 81, 100, 128 * 128]  # areas of 1 pixel


def uscene_hist(
    data,
    threshold: float = THRESHOLD,
    intensity_bins: list[float] = INTENSITY_BINS,
    area_bins: list[float] = AREA_BINS,
):
    """
    Calculate the probability mass function (PMF) of rainfall intensity and area.

    Parameters
    ----------
    data : array-like
        The input rainfall data.
    threshold : float, optional
        The threshold value to identify rainfall events. Default is THRESHOLD.
    intensity_bins : list, optional
        The bins for rainfall intensity. Default is INTENSITY_BINS.
    area_bins : list, optional
        The bins for rainfall area. Default is AREA_BINS.

    Returns
    -------
    pmf_intensity : np.ndarray
        The PMF of rainfall intensity.
    pmf_area : np.ndarray
        The PMF of rainfall area.
    """
    # Filter data based on threshold

    id_regions, num_ids = ndimage.label(data >= threshold)

    region_intensities = ndimage.maximum(
        data, id_regions, index=np.arange(0, num_ids + 1)
    )
    region_areas = ndimage.labeled_comprehension(
        data,
        id_regions,
        index=np.arange(0, num_ids + 1),
        func=len,
        out_dtype=int,
        default=0,
    )

    hist = np.histogram2d(region_areas, region_intensities, [AREA_BINS, INTENSITY_BINS])

    return hist[0]


def calc_pmf(da: xr.DataArray):
    """
    Calculate the probability mass function (PMF) of rainfall intensity and area.

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
            uscene_hist,
            da,
            input_core_dims=[[da.cf["X"].name, da.cf["Y"].name]],
            output_core_dims=[["area_bin", "intensity_bin"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={
                "output_sizes": {
                    "area_bin": len(AREA_BINS) - 1,
                    "intensity_bin": len(INTENSITY_BINS) - 1,
                }
            },
            output_dtypes=["float64"],
        )
        # .drop_isel(intensity_bin=0)  # Drop the first bin which corresponds to entirely dry intensity
        .sum(dim=["ensemble_member", "time"]).compute()
    )
    return (hist / hist.sum(dim=["area_bin", "intensity_bin"])).rename(
        "Probability Mass"
    )


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
    xticks = np.arange(0, len(INTENSITY_BINS), 1)
    yticks = np.arange(0, len(AREA_BINS), 1)
    shw = ax.pcolormesh(
        xticks,
        yticks,
        pmf,
        **kwargs,
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(INTENSITY_BINS, rotation=90, fontsize="x-small")
    ax.set_yticks(yticks)
    ax.set_yticklabels(AREA_BINS, fontsize="x-small")
    ax.set_title(title)

    return shw

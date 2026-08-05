import numpy as np
import scipy.ndimage as ndimage
import xarray as xr

# THRESHOLD = 0.01 # threshold value in mm/10min
# BINS = [0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0,1000.0]
THRESHOLD = 0.1  # threshold value in mm/hr used to mark end of a spell
INTENSITY_BINS = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1000.0]
DURATIONS_BINS = [1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 72]  # durations of 1 hour


def calc_spells(data, threshold=THRESHOLD):
    """
    Calculate spells of consecutive values above a threshold in a time series.

    Parameters
    ----------
    data : array-like
        The input time series data.
    threshold : float, optional
        The threshold value to identify spells. Default is THRESHOLD.

    Returns
    -------
    np.ndarray
        An array of spells, where each spell is represented as a tuple containing:
        (start_position, spell_length, max_value_in_spell)
    """
    data_over_threshold = data >= threshold
    start_position = 0
    spells = []
    while start_position < len(data_over_threshold):
        # find next start position in time-series area where value is above threshold
        if data_over_threshold[start_position] is False:
            start_position += 1
            continue
        current_spell_length = 1
        # keep adding 1 to spell length until the next value is below threshold or we reach the end of the data
        while all(
            data_over_threshold[
                start_position : start_position + current_spell_length + 1
            ]
        ) and start_position + current_spell_length < len(data_over_threshold):
            current_spell_length += 1
        spells.append(
            (
                start_position,
                current_spell_length,
                data[start_position : start_position + current_spell_length].max(),
            )
        )
        start_position += current_spell_length
    return spells

    # # ALTERNATIVE CODE
    # spells = []
    # for spell_length in xrange(5,0,-1):
    #    windows = rolling_window(data,window=spell_length)
    #    full_windows = np.all(windows,axis=1)
    #    start = np.where(full_windows)[0]
    #    for s in start:
    #        e = s + spell_length - 1
    #        data[s:e+1] = False
    #        spells.append([s,e])
    # return spells


def ucalc_distn(data, threshold=THRESHOLD):
    spells = calc_spells(data, threshold=threshold)
    return calc_distn(np.array(spells, dtype=np.float64))[0]


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
    # dry bin spells are lumped into one spell but should be counted as multiple spells of length 1, so we set the dry intensity, short duration bin to the "length" of the dry spells and other dry durations to 0
    hist[0][:, 0] = 0
    hist[0][0, 0] = region_durs[0]
    return hist[0]


def calc_distn(spell_data):
    """
    Calculate the 2D histogram of spell durations and maximum values.
    """

    hist = np.histogram2d(
        spell_data[:, 1], spell_data[:, 2], [DURATIONS_BINS, INTENSITY_BINS]
    )
    return hist


def calc_pmf(da):
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
            ucalc_distn,
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
        .sum(dim=["ensemble_member", "grid_latitude", "grid_longitude"])
        .compute()
    )
    return hist / hist.sum()


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
        .sum(dim=["ensemble_member", "grid_latitude", "grid_longitude"])
        .compute()
    )
    return hist / hist.sum()


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
    yticks = np.arange(0, len(DURATIONS_BINS), 1)
    shw = ax.pcolormesh(
        xticks,
        yticks,
        pmf,
        **kwargs,
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(INTENSITY_BINS, rotation=90, fontsize="x-small")
    ax.set_yticks(yticks)
    ax.set_yticklabels(DURATIONS_BINS, fontsize="x-small")
    ax.set_title(title)

    return shw

import numpy as np
import matplotlib

# THRESHOLD = 0.01 # threshold value in mm/10min
# BINS = [0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0,1000.0]
THRESHOLD = 0.05  # threshold value in mm/hr used to mark end of a spell
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
    return spells  # np.array(spells, dtype=np.float64)

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


def calc_distn(spell_data):
    """
    Calculate the 2D histogram of spell durations and maximum values.
    """

    hist = np.histogram2d(
        spell_data[:, 1], spell_data[:, 2], [DURATIONS_BINS, INTENSITY_BINS]
    )
    return hist


def calc_pmf(spell_data):
    """
    Calculate the 2D probability mass function (PMF) of spell durations and maximum values.

    Parameters
    ----------
    spell_data : array-like
        The input spell data, where each row represents a spell with its duration and maximum value.

    Returns
    -------
    np.ndarray
        A 2D array representing the PMF of spell durations and maximum values.
    """
    hist = calc_distn(np.array(spell_data, dtype=np.float64))
    pmf = hist[0] / np.sum(hist[0])
    return pmf


def plot_pmf(ax, pmf, title):
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
        xticks, yticks, pmf, norm=matplotlib.colors.LogNorm(vmin=0.001, vmax=0.1)
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(INTENSITY_BINS, rotation=90, fontsize="x-small")
    ax.set_yticks(yticks)
    ax.set_yticklabels(DURATIONS_BINS, fontsize="x-small")
    ax.set_title(title)

    return shw

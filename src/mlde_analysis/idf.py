import numpy as np

# THRESHOLD = 0.01 # threshold value in mm/10min
# BINS = [0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0,1000.0]
THRESHOLD = 0.1  # threshold value in mm/hr
BINS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1000.0]
NBINS = len(BINS) - 1
# DURATIONS = list(range(1,25))
# DURATIONS.extend([30,36,42,48,54,60,66,72,96,120,144]) # durations of 10 mins
DURATIONS = list(range(1, 7)) + [12, 18, 24, 36, 48, 72]  # durations of 1 hour
NDURS = len(DURATIONS) - 1
NYEARS = 13

SEASON_SELECT = "MAM"


def calc_spells(data, threshold=THRESHOLD):
    data_over_threshold = data >= threshold
    start_position = 0
    spells = []
    while start_position < len(data_over_threshold):
        if data_over_threshold[start_position] is False:
            start_position += 1
            continue
        current_spell_length = 1
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
    return np.array(spells, dtype=np.float64)

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


def select_spells(spell_data, season_timeseries, season_select=SEASON_SELECT):
    spells_select = []
    end_positions = spell_data[:, 0] + spell_data[:, 1] - 1
    for spell, e in enumerate(end_positions):
        if season_timeseries[int(e)] == season_select:
            spells_select.append(spell_data[spell, :])
    return np.array(spells_select)


def calc_distn(spell_data):
    hist = np.histogram2d(spell_data[:, 1], spell_data[:, 2], [DURATIONS, BINS])
    return hist

import cf_xarray  # noqa: F401
import dask
import numpy as np
import xarray as xr

from mlde_analysis.distribution import xr_hist


def stats(da: xr.DataArray, var_range: tuple) -> xr.Dataset:
    """
    Compute the cache statistics for a given dataset or sample set.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array for which to compute cache statistics.

    Returns
    -------
    xr.Dataset
        A dataset containing the computed cache statistics:
        * NaN count
        * Maximum
        * mean, standard deviation, quantiles over time and ensemble members
        * frequency density histogram
    """

    nbins = 200
    bins = np.histogram_bin_edges([], bins=nbins, range=var_range)
    hist_da, bins = xr_hist(da, bins=bins)

    stats = xr.merge(
        [
            dask.array.isnan(da).sum().rename(f"NaN Count"),
            da.max().rename(f"Max Value"),
            da.cf.mean(dim=["T", "ensemble_member"]).rename("Mean"),
            da.cf.std(dim=["T", "ensemble_member"]).rename("Standard Deviation"),
            da.cf.quantile(0.999, dim=["T", "ensemble_member"])
            .drop("quantile")
            .rename("99.9th Percentile"),
            hist_da,
        ],
        compat="no_conflicts",
    )

    return stats

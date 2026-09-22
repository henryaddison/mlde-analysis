import cf_xarray  # noqa: F401
import dask
import numpy as np
import xarray as xr

from mlde_analysis.distribution import xr_hist


def from_dataset(
    ds: xr.Dataset, var_range: tuple, variables: list[str]
) -> dict[str, xr.DataTree]:
    """
    Compute the statistics for a given Dataset for a given list of variables.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to extract dataarrays for computing statistics.
    var_range : tuple
        The range of values for the histogram.
    eval_vars : list
        A list of variable names for which to compute statistics.

    Returns
    -------
    dict[str, xr.DataTree]
        A dictionary of data trees containing the computed cache statistics:
        * NaN count
        * Maximum
        * mean, standard deviation, quantiles over time and ensemble members
        * frequency density histogram
    """

    return xr.DataTree.from_dict(
        {var: from_dataarray(ds.cf[var], var_range) for var in variables}
    )


def from_dataarray(da: xr.DataArray, var_range: tuple) -> xr.DataTree:
    """
    Compute the cache statistics for a given dataset or sample set.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array for which to compute cache statistics.

    Returns
    -------
    xr.DataTree
        A data tree containing the computed cache statistics:
        * NaN count
        * Maximum
        * mean, standard deviation, quantiles over time and ensemble members
        * frequency density histogram
    for whole array and for day quarter and seasonal groupings.
    """

    nbins = 200

    root_stats = _basic_stats(da, nbins=nbins, var_range=var_range)

    day_qtr_stats = da.groupby_bins("time.hour", [-1, 5, 11, 17, 23]).map(
        _basic_stats, nbins=nbins, var_range=var_range
    )

    # replace hour_bins with bounds that can be saved to zarr
    # groupby_bins uses Interval objects which cannot be saved to zarr
    hr_bnds = xr.DataArray(
        data=np.stack(
            [
                day_qtr_stats["hour_bins"].data.left.values,
                day_qtr_stats["hour_bins"].data.right.values,
            ],
            axis=1,
        ),
        dims=["hour_bins", "bnds"],
        name="hour_bins_bnds",
    )

    day_qtr_stats["hour_bins"] = day_qtr_stats["hour_bins"].data.mid
    day_qtr_stats = xr.merge([day_qtr_stats, hr_bnds]).drop_attrs()

    seasonal_stats = da.groupby("time.season").map(
        _basic_stats, nbins=nbins, var_range=var_range
    )

    tree = xr.DataTree(
        dataset=root_stats,
        children={
            "day_qrt": xr.DataTree(dataset=day_qtr_stats),
            "seasonal": xr.DataTree(dataset=seasonal_stats),
        },
    )

    return tree


def _basic_stats(da: xr.DataArray, nbins: int, var_range: tuple) -> xr.Dataset:
    bins = np.histogram_bin_edges([], bins=nbins, range=var_range)
    hist_da, bins = xr_hist(da, bins=bins)

    return xr.merge(
        [
            dask.array.isnan(da).sum().rename(f"NaN Count"),
            da.max().rename(f"max"),
            da.cf.mean(dim=["T", "ensemble_member"]).rename("mean"),
            da.cf.std(dim=["T", "ensemble_member"]).rename("std"),
            da.cf.quantile(0.999, dim=["T", "ensemble_member"])
            .drop("quantile")
            .rename("q999"),
            da.cf.where(da > 60).count().rename("vhi_exceedence_count"),
            hist_da,
        ],
        compat="no_conflicts",
    )

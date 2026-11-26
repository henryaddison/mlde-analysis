import numpy as np
import xarray as xr

from mlde_analysis import stats


def test_basic_stats_keeps_exact_quantiles_lazy():
    da = xr.DataArray(
        np.arange(120, dtype=float).reshape(4, 30),
        dims=("ensemble_member", "time"),
        coords={
            "time": xr.DataArray(
                np.arange(
                    np.datetime64("2000-01-01T00"),
                    np.datetime64("2000-01-02T06"),
                    np.timedelta64(1, "h"),
                ),
                dims="time",
                attrs={"axis": "T"},
            )
        },
    ).chunk({"ensemble_member": 2, "time": 10})

    actual = stats._basic_stats(da, nbins=20, var_range=(0, 200))

    for quantile, name in [
        (0.999, "q_0.999"),
        (0.9999, "q_0.9999"),
        (0.99999, "q_0.99999"),
    ]:
        assert hasattr(actual[name].data, "__dask_graph__")
        expected = da.cf.quantile(quantile, dim=["T", "ensemble_member"]).drop_vars(
            "quantile"
        )
        xr.testing.assert_allclose(actual[name].compute(), expected.compute())

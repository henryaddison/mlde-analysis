from statsmodels.tsa.stattools import acf
import xarray as xr


def full_dask_acf(da, nlags=23):
    return xr.apply_ufunc(
        acf,
        da.squeeze("date"),
        input_core_dims=[["hour"]],
        output_core_dims=[["lag"]],
        vectorize=True,
        kwargs=dict(nlags=nlags),
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                "lag": nlags + 1,
            }
        },
    )


def midnight_jump_correlations(da):
    around_jump_ts_da = xr.concat(
        [
            xr.concat(
                [
                    sda.shift(time=2)
                    .isel(time=slice(0, -1, 24))
                    .expand_dims(hours_after_midnight=[-2]),
                    sda.shift(time=1)
                    .isel(time=slice(0, -1, 24))
                    .expand_dims(hours_after_midnight=[-1]),
                    sda.shift(time=0)
                    .isel(time=slice(0, -1, 24))
                    .expand_dims(hours_after_midnight=[0]),
                    sda.shift(time=-1)
                    .isel(time=slice(0, -1, 24))
                    .expand_dims(hours_after_midnight=[1]),
                ],
                dim="hours_after_midnight",
                join="exact",
            ).isel(time=slice(1, None))
            for season_start, sda in da.groupby(
                time=xr.groupers.SeasonResampler(["DJF", "MAM", "JJA", "SON"])
            )
        ],
        dim="time",
    ).compute()

    around_jump_corrs = xr.concat(
        [
            xr.corr(
                around_jump_ts_da.sel(hours_after_midnight=i - 1),
                around_jump_ts_da.sel(hours_after_midnight=i),
                dim=["time", "ensemble_member"],
            ).expand_dims(hours_after_jump=[i])
            for i in [-1, 0, 1]
        ],
        dim="hours_after_jump",
    )

    return around_jump_corrs

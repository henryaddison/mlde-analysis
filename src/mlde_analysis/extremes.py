import numpy as np
import xarray as xr

from .display import pretty_table


def return_levels(da, n_days_per_year):
    # Make axis for return times; nt = number of time points; n_days_per_year is no. of days in year (the no. in one season if considering just winter or summer etc.)
    da = da.stack(example=["time", "ensemble_member"])
    nt = len(da["example"])
    num_years = nt / n_days_per_year
    return_times_axis = xr.DataArray(
        data=num_years / (np.arange(nt)[::-1] + 1),
        dims=["rp"],
        attrs={"long_name": "Return period", "units": "Year"},
    )

    return xr.DataArray(
        data=np.ma.sort(np.squeeze(da.values), axis=None),
        dims=["rp"],
        coords={"rp": return_times_axis},
        attrs=da.attrs,
    )


def pred_and_target_return_levels(ds, var, n_days_per_year=360):
    return xr.merge(
        [
            ds[f"pred_{var}"]
            .groupby("sample_id")
            .map(
                lambda da: da.groupby("model").map(
                    return_levels, n_days_per_year=n_days_per_year
                )
            )
            .rename(f"pred_{var}_return_level"),
            return_levels(
                ds[f"target_{var}"],
                n_days_per_year=n_days_per_year,
            ).rename(f"target_{var}_return_level"),
        ]
    )


def plot_return_levels(pred_rt_da, cpm_rt_da, row=None):

    g = pred_rt_da.plot(
        x="rp", col="model", row=row, hue="sample_id", marker="+", alpha=0.5
    )
    if row:
        for d, ax in zip(g.name_dicts.flat, g.axs.flat, strict=True):
            # None is the sentinel value
            if d is not None:
                row_cpm_rt_da = cpm_rt_da.sel({row: d[row]})
                ax.plot(
                    row_cpm_rt_da["rp"],
                    row_cpm_rt_da,
                    label="CPM",
                    color="k",
                    linestyle="--",
                    marker="x",
                    zorder=-100,
                )
    else:
        for ax in g.axs.flat:
            ax.plot(
                cpm_rt_da["rp"],
                cpm_rt_da.squeeze(row),
                label="CPM",
                color="k",
                linestyle="--",
                marker="x",
                zorder=-100,
            )

    for ax in g.axs.flat:
        ax.set_xscale("log")
        if cpm_rt_da["rp"].max() > 100:
            ax.set_xlim((1, None))


def pretty_return_levels_table(rt_ds, var, rps=[1, 10, 100]):
    rl_table_ds = rt_ds.sel(rp=rps, method="nearest")
    pred_rl_da = rl_table_ds[f"pred_{var}_return_level"]
    target_rl_da = rl_table_ds[f"target_{var}_return_level"]
    rl_errs_da = (pred_rl_da - target_rl_da).rename(f"{var} Return level errors")
    rl_errs_da.values.sort(axis=-2)
    rl_pcerrs_da = (rl_errs_da / target_rl_da * 100).rename(
        f"{var} Return level percent errors (%)"
    )

    pretty_table(
        xr.merge([rl_errs_da, rl_pcerrs_da]),
        round=3,
        pivot_table=dict(
            index=["location", "model", "rp"],
            columns="sample_id",
            values=[
                f"{var} Return level errors",
                f"{var} Return level percent errors (%)",
            ],
        ),
    )

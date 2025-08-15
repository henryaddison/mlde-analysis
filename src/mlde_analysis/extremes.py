import numpy as np
import xarray as xr


def return_time_amounts(da, n_days_per_year):
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


def pred_and_target_return_times(ds, var, n_days_per_year=360):
    return xr.merge(
        [
            ds[f"pred_{var}"]
            .groupby("sample_id")
            .map(
                lambda da: da.groupby("model").map(
                    return_time_amounts, n_days_per_year=n_days_per_year
                )
            )
            .rename(f"pred_{var}_return_level"),
            return_time_amounts(
                ds[f"target_{var}"],
                n_days_per_year=n_days_per_year,
            ).rename(f"target_{var}_return_level"),
        ]
    )


def plot_return_time_amounts(pred_rt_da, cpm_rt_da, row=None):

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

import numpy as np
import xarray as xr


def ndays_threshold_exceeded_in_area(da, threshold):
    return (da.max(dim=["grid_longitude", "grid_latitude"]) > threshold).sum(
        dim="example"
    )


def return_time_at_threshold(da, threshold):
    total_days = len(da["example"])
    return total_days / ndays_threshold_exceeded_in_area(da, threshold)


def return_times(da, thresholds, var):
    da = da.stack(example=["time", "ensemble_member"])

    rt_da = xr.concat(
        map(
            lambda thr: return_time_at_threshold(da, thr),
            thresholds,
        ),
        dim=var,
    ).assign_coords({var: thresholds})
    rt_da[var] = rt_da[var].assign_attrs(da.attrs)

    return rt_da


def plot_return_times(pred_rt_da, cpm_rt_da, var):
    g = pred_rt_da.plot(y=var, col="model", hue="sample_id", marker="+")
    for ax in g.axs.flat:
        ax.plot(
            cpm_rt_da,
            cpm_rt_da[var],
            label="cpm",
            color="k",
            linestyle="--",
            marker="x",
        )
        ax.set_xscale("log")
        if cpm_rt_da.max() > 100:
            ax.set_xlim((1, None))


def return_time_amounts(da, n_days_per_year=360):
    # Make axis for return times; nt = number of time points; n_days_per_year is no. of days in year (the no. in one season if considering just winter or summer etc.)
    da = da.stack(example=["time", "ensemble_member"])
    nt = len(da["example"])
    return_times_axis = float(nt) / n_days_per_year / (np.arange(nt)[::-1] + 1)

    rt_da = xr.DataArray(
        data=(np.ma.sort(da.values, axis=None)),
        dims=["rp"],
        coords={"rp": return_times_axis},
        attrs=da.attrs,
    )

    rt_da["rp"] = rt_da["rp"].assign_attrs(
        {"long_name": "Return time", "units": "Year"}
    )

    return rt_da


def plot_return_time_amounts(pred_rt_da, cpm_rt_da):
    g = pred_rt_da.plot(x="rp", col="model", hue="sample_id", marker="+")
    for ax in g.axs.flat:
        ax.plot(
            cpm_rt_da["rp"],
            cpm_rt_da,
            label="cpm",
            color="k",
            linestyle="--",
            marker="x",
        )
        ax.set_xscale("log")
        if cpm_rt_da["rp"].max() > 100:
            ax.set_xlim((1, None))

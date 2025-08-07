import xarray as xr


def ndays_threshold_exceeded_in_area(da, threshold):
    return ((da > threshold).sum(dim=["grid_longitude", "grid_latitude"]) > 0).sum(
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
    g = pred_rt_da.plot(y=var, col="model", hue="sample_id")
    for ax in g.axs.flat:
        ax.plot(cpm_rt_da, cpm_rt_da[var], label="cpm", color="k", linestyle="--")
        ax.set_xscale("log")

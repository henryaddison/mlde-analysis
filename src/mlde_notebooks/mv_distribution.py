import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def xr_hist2d(x, y, bins, **kwargs):
    def _np_hist2d(x, y, bins, **kwargs):
        return np.histogram2d(
            x=x.flatten(), y=y.flatten(), bins=bins, density=True, **kwargs
        )[0]

    return xr.apply_ufunc(
        _np_hist2d,
        x,
        y,
        input_core_dims=[x.dims, y.dims],
        output_core_dims=[["xbins", "ybins"]],
        vectorize=True,
        kwargs=kwargs | dict(bins=bins),
    ).assign_attrs(
        xlabel=xr.plot.utils.label_from_attrs(da=x),
        ylabel=xr.plot.utils.label_from_attrs(da=y),
    )


def compute_hist2d(x_pred, y_pred, x_target, y_target, xbins, ybins):
    print(x_target.count())
    target_hist = xr.merge(
        [
            xr_hist2d(x_target, y_target, bins=(xbins, ybins)).rename(
                "target_2d_density"
            ),
            x_target.count().rename("xcount"),
            y_target.count().rename("ycount"),
        ]
    )

    pred_hist = (
        xr.merge([x_pred.rename("x_pred"), y_pred.rename("y_pred")])
        .groupby("model")
        .map(lambda ds: xr_hist2d(ds["x_pred"], ds["y_pred"], bins=(xbins, ybins)))
        .rename("pred_2d_density")
    )

    return xr.merge([target_hist, pred_hist])


def plot_hist2d_figure(hist2d_ds, xbins, ybins, cutoff=True):
    xlabel = hist2d_ds.attrs["xlabel"]
    ylabel = hist2d_ds.attrs["ylabel"]
    # density of a bin with just a single value from a target examples
    # NB assumes all bins are same size
    count = hist2d_ds["xcount"].values.item()
    single_value_target_density = 1 / (
        (xbins[1] - xbins[0]) * (ybins[1] - ybins[0]) * count
    )
    # make the smallest density we care about in viz slightly smaller
    density_cutoff = 0.99 * single_value_target_density
    print(count)
    print(density_cutoff)

    fig = plt.figure(layout="constrained", figsize=(9, 6))
    grid_spec = [
        ["target", *(hist2d_ds["model"].values)],
        [".", *(hist2d_ds["model"].values + " diff")],
    ]

    axd = fig.subplot_mosaic(grid_spec)  # , sharex=True, sharey=True)

    norm = matplotlib.colors.LogNorm(
        vmin=density_cutoff,
        vmax=max(
            [hist2d_ds["target_2d_density"].max(), hist2d_ds["pred_2d_density"].max()]
        ),
    )
    cmap = matplotlib.colormaps["viridis"].resampled(9)
    if cutoff:
        cmap.set_under("white", alpha=0)
    cf = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax = axd["target"]

    ax.pcolormesh(
        xbins, ybins, hist2d_ds["target_2d_density"].T, cmap=cf.cmap, norm=cf.norm
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("CPM")

    for model, model_pred_hist2d_da in hist2d_ds["pred_2d_density"].groupby("model"):
        ax = axd[model]

        ax.pcolormesh(xbins, ybins, model_pred_hist2d_da.T, cmap=cf.cmap, norm=cf.norm)
        ax.set_xlabel(xlabel)
        ax.set_title(model)

    fig.colorbar(cf, ax=[axd[s] for s in grid_spec[0]])

    diff_da = np.abs(hist2d_ds["pred_2d_density"] - hist2d_ds["target_2d_density"])

    norm = matplotlib.colors.LogNorm(
        vmin=density_cutoff,
        vmax=diff_da.max().values.item(),
    )
    cmap = matplotlib.colormaps["turbo"].resampled(9)
    if cutoff:
        cmap.set_under("white", alpha=0)
    cf = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

    for model, model_diff_da in diff_da.groupby("model"):
        ax = axd[f"{model} diff"]

        ax.pcolormesh(xbins, ybins, model_diff_da.T, cmap=cf.cmap, norm=cf.norm)
        ax.set_xlabel(xlabel)
        ax.set_title(f"|{model} - CPM|")

    fig.colorbar(cf, ax=[axd[s] for s in grid_spec[1] if s != "."])

    return fig, axd

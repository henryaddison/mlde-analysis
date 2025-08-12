import numpy as np
import scipy
import xarray as xr


def plot_domain_means(pred_da, target_da, ax, line_props, alpha=0.05):
    pred_mean_da = pred_da.mean(dim=["grid_latitude", "grid_longitude"]).assign_attrs(
        pred_da.attrs
    )
    target_mean_da = target_da.mean(
        dim=["grid_latitude", "grid_longitude"]
    ).assign_attrs(target_da.attrs)

    ax.plot(
        target_mean_da.broadcast_like(pred_mean_da).values.flat,
        pred_mean_da.values.flat,  # .squeeze("model"),
        alpha=alpha,
        color=line_props["color"],
        marker=".",
        markersize=3,
        linewidth=0,
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(
        [0, 1],
        [0, 1],
        transform=ax.transAxes,
        linewidth=1,
        color="black",
        linestyle="--",
        label="Ideal",
    )

    ax.set_title(line_props["label"], fontsize="medium")
    ax.set_xlabel(
        f"CPM mean\n{xr.plot.utils.label_from_attrs(da=target_mean_da)}",
        fontsize="small",
    )
    ax.set_ylabel(
        f"ML mean\n{xr.plot.utils.label_from_attrs(da=pred_mean_da)}",
        fontsize="small",
    )


def _corrected_ensemble_variance(pred_da):
    """
    Corrects the ensemble spread for the finite ensemble size
    """
    # Need to correct for the finite ensemble (or num samples runs) size of samples
    # Equation 7 from # Leutbecher, M., & Palmer, T. N. (2008). Ensemble forecasting. Journal of Computational Physics, 227(7), 3515-3539. doi:10.1016/j.jcp.2007.02.014

    ensemble_size = len(pred_da["sample_id"])
    variance_correction_term = (ensemble_size + 1) / (ensemble_size - 1)

    return variance_correction_term * np.power(
        pred_da - pred_da.mean(dim="sample_id"), 2
    ).mean(dim="sample_id")


def se_bins(pred_da, target_da, nbins=100):
    """
    For an "ensemble" of predicted rainfall and the corresponding "truth" value,
    this computes bins for spread and error over the whole dataset
    with the aim of re-using common bins for subsets of the data
    """
    ensemble_variance = _corrected_ensemble_variance(pred_da).values.flatten()
    bins = np.nanquantile(ensemble_variance, np.linspace(0, 1, nbins + 1))
    # remove bin edges too near each other
    bins = np.delete(bins, np.argwhere(np.ediff1d(bins) <= 1e-6) + 1)

    return bins


def compute_rmss_rmse_bins(pred_da, target_da, bins):
    """
    For an "ensemble" of predicted rainfall and the coresponding "truth" value,
    this computes bins for spread and error for a spread-error plot

    Sources:

    * https://journals.ametsoc.org/view/journals/hydr/15/4/jhm-d-14-0008_1.xml?tab_body=fulltext-display
    * https://journals.ametsoc.org/view/journals/aies/2/2/AIES-D-22-0061.1.xml
    * https://www.sciencedirect.com/science/article/pii/S0021999107000812
    """

    squared_error = np.power(
        pred_da.mean(dim=["sample_id"]) - target_da, 2
    ).values.flatten()
    ensemble_variance = _corrected_ensemble_variance(pred_da).values.flatten()

    # remove NaNs
    not_nans = ~np.isnan(squared_error)
    squared_error = squared_error[not_nans]
    ensemble_variance = ensemble_variance[not_nans]

    if isinstance(bins, int):
        bins = se_bins(pred_da, target_da, nbins=bins)

    spread_binned_mse, _, abinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, squared_error, statistic="mean", bins=bins
    )
    spread_binned_rmse = np.sqrt(spread_binned_mse)

    spread_binned_mss, _, bbinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, ensemble_variance, statistic="mean", bins=bins
    )
    spread_binned_rmss = np.sqrt(spread_binned_mss)

    assert (abinnumbers == bbinnumbers).all()

    bin_counts = np.bincount(bbinnumbers)[1:]  # [1:] as binnumbers start at 1

    assert bin_counts.sum() == len(ensemble_variance)

    ssrel = (np.abs(spread_binned_rmse - spread_binned_rmss) * bin_counts).sum() / len(
        ensemble_variance
    )

    serat = np.sqrt(ensemble_variance.mean()) / np.sqrt(squared_error.mean())
    # import pdb; pdb.set_trace()
    return xr.Dataset(
        {
            "spread_binned_rmss": (
                ["bin"],
                spread_binned_rmss,
                {"units": target_da.attrs.get("units", "")},
            ),
            "spread_binned_rmse": (
                ["bin"],
                spread_binned_rmse,
                {"units": target_da.attrs.get("units", "")},
            ),
            "ssrel": ([], ssrel),
            "serat": ([], serat),
            "bin_edges": (
                ["bin", "bnds"],
                np.concatenate(
                    (bins[:-1].reshape(-1, 1), bins[1:].reshape(-1, 1)), axis=1
                ),
            ),
        },
        coords={"bin": bins[1:]},
    )


def plot_spread_error(spread_error_ds, ax, line_props, bs_dim=None):
    for model, model_spread_error_ds in spread_error_ds.groupby("model"):
        model_spread_error_ds = model_spread_error_ds.squeeze("model")
        if bs_dim is None:
            x = model_spread_error_ds["spread_binned_rmss"]
            y = model_spread_error_ds["spread_binned_rmse"]
            ax.plot(
                x,
                y,
                label=f"{model}",
                color=line_props[model]["color"],
                marker=".",
                alpha=0.25,
                markersize=3,
                linewidth=0,
            )
        else:
            x = model_spread_error_ds["spread_binned_rmss"].mean(dim=bs_dim)
            y = model_spread_error_ds["spread_binned_rmse"].mean(dim=bs_dim)
            xerr = (
                x
                - model_spread_error_ds["spread_binned_rmss"].quantile(
                    dim=bs_dim, q=0.05
                ),
                model_spread_error_ds["spread_binned_rmss"].quantile(dim=bs_dim, q=0.95)
                - x,
            )
            yerr = (
                y
                - model_spread_error_ds["spread_binned_rmse"].quantile(
                    dim=bs_dim, q=0.05
                ),
                model_spread_error_ds["spread_binned_rmse"].quantile(dim=bs_dim, q=0.95)
                - y,
            )

            ax.errorbar(
                x=x,
                y=y,
                xerr=xerr,
                yerr=yerr,
                label=f"{model}",
                color=line_props[model]["color"],
                marker=".",
                alpha=0.25,
                markersize=3,
                linewidth=0,
                elinewidth=0.5,
                ecolor="k",
            )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(
        [0, 1],
        [0, 1],
        transform=ax.transAxes,
        label="ideal",
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    # ax.legend()

    ax.set_xlabel(
        f"RMSS {spread_error_ds["spread_binned_rmss"].attrs.get('units', '')}",
        fontsize="small",
    )
    ax.set_ylabel(
        f"RMSE {spread_error_ds["spread_binned_rmse"].attrs.get('units', '')}",
        fontsize="small",
    )
    ax.set_title("CPM Diffusion\nSpread-Error", fontsize="medium")

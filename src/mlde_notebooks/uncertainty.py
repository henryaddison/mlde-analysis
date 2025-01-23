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


def compute_rmss_rmse_bins(pred_pr, target_pr, nbins=100):
    """
    For an "ensemble" of predicted rainfall and the coresponding "truth" value,
    this computes bins for spread and error for a spread-error plot

    Sources:

    * https://journals.ametsoc.org/view/journals/hydr/15/4/jhm-d-14-0008_1.xml?tab_body=fulltext-display
    * https://journals.ametsoc.org/view/journals/aies/2/2/AIES-D-22-0061.1.xml
    * https://www.sciencedirect.com/science/article/pii/S0021999107000812
    """

    # Need to correct for the finite ensemble (or num samples runs) size of samples
    # Equation 7 from # Leutbecher, M., & Palmer, T. N. (2008). Ensemble forecasting. Journal of Computational Physics, 227(7), 3515-3539. doi:10.1016/j.jcp.2007.02.014
    ensemble_size = len(pred_pr["sample_id"])
    variance_correction_term = (ensemble_size + 1) / (ensemble_size - 1)

    ensemble_mean = pred_pr.mean(dim=["sample_id"])
    ensemble_variance = (
        variance_correction_term
        * np.power(pred_pr - ensemble_mean, 2).mean(dim="sample_id").values.flatten()
    )

    squared_error = np.power(ensemble_mean - target_pr, 2).values.flatten()

    bin_edges = np.quantile(ensemble_variance, np.linspace(0, 1, nbins + 1))
    # remove bin edges too near each other
    bin_edges = np.delete(bin_edges, np.argwhere(np.ediff1d(bin_edges) <= 1e-6) + 1)

    spread_binned_mse, _, abinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, squared_error, statistic="mean", bins=bin_edges
    )
    spread_binned_rmse = np.sqrt(spread_binned_mse)

    spread_binned_variance, _, bbinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, ensemble_variance, statistic="mean", bins=bin_edges
    )
    spread_binned_rmss = np.sqrt(spread_binned_variance)

    assert (abinnumbers == bbinnumbers).all()

    return spread_binned_rmss, spread_binned_rmse


def plot_spread_error(pred_da, target_da, ax, line_props):

    for model, model_pred_da in pred_da.groupby("model"):

        binned_rmss, binned_rmse = compute_rmss_rmse_bins(model_pred_da, target_da)

        ax.plot(
            binned_rmss,
            binned_rmse,
            label=f"{model}",
            color=line_props[model]["color"],
            marker=".",
            alpha=0.25,
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
        label="ideal",
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    # ax.legend()

    ax.set_xlabel(f"RMSS {target_da.attrs.get('units', '')}", fontsize="small")
    ax.set_ylabel(f"RMSE {target_da.attrs.get('units', '')}", fontsize="small")
    ax.set_title("CPM Diffusion\nSpread-Error", fontsize="medium")

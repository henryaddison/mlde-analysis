import matplotlib.pyplot as plt
import matplotlib
import xarray as xr

from mlde_utils import cp_model_rotated_pole


def compute_correlations(x_pred, y_pred, x_target, y_target, corr_f=xr.corr):
    target_corr = corr_f(
        x_target,
        y_target,
        dim=[
            "ensemble_member",
            "time",
        ],
    ).rename("Target Corr")
    pred_corr = corr_f(
        x_pred, y_pred, dim=["ensemble_member", "time", "sample_id"]
    ).rename("Pred Corr")
    corr_diff = (pred_corr - target_corr).rename("Corr diff")
    return xr.merge(
        [
            target_corr,
            pred_corr,
            corr_diff,
        ]
    )


def plot_correlations(corr_ds, label="Corr"):
    target_corr = corr_ds["Target Corr"]
    pred_corr = corr_ds["Pred Corr"]
    corr_diff = corr_ds["Corr diff"]
    mosaic = [
        ["CPM"] + list(pred_corr["model"].values),
        ["."] + [f"{model} diff" for model in pred_corr["model"].values],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic,
        sharex=True,
        sharey=True,
        figsize=(4.5, 3),
        subplot_kw=dict(projection=cp_model_rotated_pole),
        constrained_layout=True,
    )
    ax = axd["CPM"]
    cmap = matplotlib.colormaps["RdBu"].reversed().resampled(9)
    target_corr.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, add_colorbar=False)
    ax.coastlines()
    ax.set_title("CPM", fontsize="small")

    for model, model_corr in pred_corr.groupby("model"):
        ax = axd[model]
        pc = model_corr.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, add_colorbar=False)
        ax.coastlines()
        ax.set_title(model, fontsize="small")

    for model, model_diff in corr_diff.groupby("model"):
        ax = axd[f"{model} diff"]
        diff_pc = model_diff.plot(
            ax=ax, cmap=cmap, vmin=-0.5, vmax=0.5, add_colorbar=False
        )
        ax.coastlines()
        ax.set_title(f"{model} - CPM", fontsize="small")

    fig.colorbar(pc, ax=[axd[axkey] for axkey in mosaic[0]], label=label)
    fig.colorbar(
        diff_pc,
        ax=[axd[axkey] for axkey in mosaic[1] if axkey != "."],
        # shrink=1.25,
        label=f"{label} diff",
    )

    return fig, axd

from matplotlib import animation
from mlde_analysis import projection_from_da, plot_map
from IPython.display import HTML, display_html
import matplotlib.pyplot as plt


def plot_frame(da, frame, ax, model_label):
    frame_da = da.isel(time=frame)
    # import pdb; pdb.set_trace()
    pr_quad = plot_map(frame_da, ax=ax, style="pr_hourly")
    ax.set_title(f"{model_label} {frame_da["time"].item()}")

    return pr_quad


def animate_day(
    ds,
    example_spec,
    example_label,
    simulation_label,
    var="pr",
    interval=500,
    filepath=None,
):
    example = (
        ds.sel(time=slice(*example_spec["times"]), **example_spec["query"])
        .isel(sample_id=0)
        .compute()
    )
    projection = projection_from_da(example.cf[f"target_{var}"])
    subplot_kw = dict(projection=projection)
    fig = plt.figure(
        figsize=(1 + 3 * (1 + example["model"].size), 3.5), layout="constrained"
    )
    mosaic = [list(example["model"].values) + ["Simulation"]]
    axd = fig.subplot_mosaic(mosaic, subplot_kw=subplot_kw)

    pr_quads = []
    for model, model_example_da in example[f"pred_{var}"].groupby("model"):
        pr_quad = plot_frame(
            model_example_da.squeeze("model"), 0, axd[model], model_label=model
        )
        pr_quads.append(pr_quad)

    pr_quad = plot_frame(
        example[f"target_{var}"], 0, axd["Simulation"], model_label=simulation_label
    )
    pr_quads.append(pr_quad)

    cb = fig.colorbar(
        pr_quads[0],
        ax=axd.values(),
        location="bottom",
        orientation="horizontal",
        shrink=0.8,
        extend="both",
    )
    cb.ax.tick_params(axis="both", which="major", labelsize="small")
    cb.set_label("Precip [mm/hr]", fontsize="small")

    fig.supylabel(example_label)

    def update(frame):
        updated_pr_quads = []

        for model, model_example_da in example[f"pred_{var}"].groupby("model"):
            pr_quad = plot_frame(
                model_example_da.squeeze("model"), frame, axd[model], model_label=model
            )
            updated_pr_quads.append(pr_quad)

        pr_quad = plot_frame(
            example[f"target_{var}"],
            frame,
            axd["Simulation"],
            model_label=simulation_label,
        )
        updated_pr_quads.append(pr_quad)

        return updated_pr_quads

    nframes = example["time"].size
    anim = animation.FuncAnimation(
        fig, update, frames=nframes, blit=False, interval=interval
    )

    if filepath is not None:
        anim.save(filepath, writer="pillow", fps=int(1000 / interval))
        print(f"Animation saved as '{filepath}'")

    plt.close(fig)  # prevent double display in notebook
    # Display as HTML5 video (works in JupyterLab)
    display_html(HTML(anim.to_html5_video()))

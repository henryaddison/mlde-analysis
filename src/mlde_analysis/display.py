import IPython
import xarray as xr

VAR_RANGES = {
    "pr": (0, 250),
    "tmean150cm": (250, 315),
    "relhum150cm": (10, 120),
    "swbgt": (-10, 40),
}

ATTRS = {
    "pr": {"long_name": "Precip.", "units": "mm/day"},
    "tmean150cm": {"long_name": "Temp.", "units": "K"},
    "relhum150cm": {"long_name": "Rel. Humidity", "units": "%"},
    "swbgt": {"long_name": "sWBGT", "units": "C"},
}


def pretty_table(
    da: xr.DataArray,
    round: int = 1,
    caption: str = None,
    dim_order: bool = None,
    render: bool = True,
) -> None:
    df = da.to_dataframe(dim_order=dim_order)
    style = df.style
    style = style.format(precision=round)
    style = style.set_table_attributes("style='display:inline'")
    if caption is not None:
        style = style.set_caption(caption)
    if render:
        IPython.display.display_html(style.to_html(), raw=True)
    return style.to_html()

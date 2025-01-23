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


def pretty_table(da: xr.DataArray, round: int = 1, dim_order=None) -> None:
    df = da.to_dataframe(dim_order=dim_order)
    df = df.style.format(precision=round)
    IPython.display.display_html(df.to_html(), raw=True)

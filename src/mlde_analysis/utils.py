from typing import Callable, List
import xarray as xr


def chained_groupby_map(da: xr.DataArray, groups: List[str], func: Callable, **kwargs):
    if len(groups) == 0:
        return func(da, **kwargs)
    else:
        return da.groupby(groups[0]).map(
            lambda gda: chained_groupby_map(gda, groups[1:], func, **kwargs)
        )

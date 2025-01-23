import cftime
import datetime
import numpy as np
import pytest
from typing import Callable
import xarray as xr


@pytest.fixture
def time(start_year: int = 1980, time_len: int = 10):
    return xr.Variable(
        ["time"],
        xr.cftime_range(
            cftime.Datetime360Day(start_year, 12, 1, 12, 0, 0, 0, has_year_zero=True),
            periods=time_len,
            freq="D",
        ),
    )


@pytest.fixture
def time_bnds(time):
    bnd_start = time.values[0] - datetime.timedelta(hours=12)
    time_bnds_values = xr.cftime_range(
        bnd_start,
        periods=len(time) + 1,
        freq="D",
    ).values
    time_bnds_pairs = np.concatenate(
        [time_bnds_values[:-1, np.newaxis], time_bnds_values[1:, np.newaxis]], axis=1
    )

    return xr.Variable(["time", "bnds"], time_bnds_pairs, attrs={})


@pytest.fixture
def grid_latitude():
    return xr.Variable(["grid_latitude"], np.linspace(-3, 3, 13), attrs={})


@pytest.fixture
def grid_longitude():
    return xr.Variable(["grid_longitude"], np.linspace(-4, 4, 17), attrs={})


@pytest.fixture
def ensemble_member():
    return xr.Variable(["ensemble_member"], np.array([f"{i:02}" for i in range(3)]))


@pytest.fixture
def dataset_coords(time, grid_longitude, grid_latitude, ensemble_member):
    return {
        "ensemble_member": ensemble_member,
        "time": time,
        "grid_longitude": grid_longitude,
        "grid_latitude": grid_latitude,
    }


@pytest.fixture
def pred_coords(time, grid_longitude, grid_latitude, ensemble_member):
    return {
        "ensemble_member": ensemble_member,
        "time": time,
        "grid_longitude": grid_longitude,
        "grid_latitude": grid_latitude,
        "sample_id": np.arange(3),
    }


def default_data_factory(shape):
    return np.random.rand(*shape)


@pytest.fixture
def variable_factory(dataset_coords):
    """Create a factory function for creating dummy xarray Variables that look like the training data."""

    def _variable_factory(
        coords=dataset_coords, data_factory=default_data_factory
    ) -> xr.Variable:
        shape = [len(c) for c in coords.values()]

        return xr.Variable(
            list(coords.keys()),
            data_factory(shape),
        )

    return _variable_factory


@pytest.fixture
def dataset_factory(
    dataset_coords, time_bnds, variable_factory
) -> Callable[[dict], xr.Dataset]:
    """Create a factory function for creating dummy xarray Datasets that look like the training data."""

    def _dataset_factory(data_vars=None) -> xr.Dataset:
        if data_vars is None:
            data_vars = {
                var: variable_factory(dataset_coords) for var in ["linpr", "target_pr"]
            }
            data_vars["target_relhum"] = variable_factory(
                lambda shape: np.random.rand(*shape) * 100
            )
        data_vars["time_bnds"] = time_bnds

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dataset_coords,
        )

        return ds

    return _dataset_factory


@pytest.fixture
def dataset(dataset_factory) -> xr.Dataset:
    """Create a dummy xarray Dataset representing a split of a set of data for training and sampling."""
    return dataset_factory()


@pytest.fixture
def predset_factory(
    pred_coords, time_bnds, variable_factory
) -> Callable[[dict], xr.Dataset]:
    """Create a factory function for creating dummy xarray Datasets that look like a set of predictions."""

    def _predset_factory(data_vars) -> xr.Dataset:
        data_vars["time_bnds"] = time_bnds

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=pred_coords,
        )

        return ds

    return _predset_factory

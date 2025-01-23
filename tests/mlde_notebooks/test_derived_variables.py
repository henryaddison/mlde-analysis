import math
import numpy as np
import pytest

from mlde_notebooks import derived_variables


@pytest.fixture
def temp_rh_ds(dataset_factory, variable_factory):
    data_vars = {
        "temp": variable_factory(
            data_factory=lambda shape: np.random.randn(*shape) + 273
        ),
        "rh": variable_factory(data_factory=lambda shape: np.random.rand(*shape) * 100),
    }
    return dataset_factory(data_vars)


def test_swbgt(temp_rh_ds):
    heat_index = derived_variables.swbgt(temp_rh_ds["temp"], temp_rh_ds["rh"])
    assert heat_index.shape == temp_rh_ds["temp"].shape

    actual = heat_index.values[1, 2, 3, 4].item()

    t = temp_rh_ds["temp"].values[1, 2, 3, 4].item()
    rh = temp_rh_ds["rh"].values[1, 2, 3, 4].item()
    tc = t - 273.15
    svp = 6.112 * math.exp((17.67 * tc) / (tc + 243.5))
    vp = (rh / 100.0) * svp
    expected = 0.567 * tc + 0.393 * vp + 3.94

    assert math.isclose(expected, actual, rel_tol=1e-5)

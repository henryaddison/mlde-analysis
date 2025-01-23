import numpy as np
import pytest
import xarray as xr

import mlde_notebooks
from mlde_notebooks import data


@pytest.fixture
def eval_dataset(dataset_factory, variable_factory) -> xr.Dataset:
    data_vars = {
        "target_temp150cm": variable_factory(
            data_factory=lambda shape: np.random.randn(*shape) + 273
        ),
        "target_relhum150cm": variable_factory(
            data_factory=lambda shape: np.random.rand(*shape) * 100
        ),
    }
    return dataset_factory(data_vars)


@pytest.fixture
def predictions(pred_coords, predset_factory, variable_factory) -> xr.Dataset:
    data_vars = {
        "pred_temp150cm": variable_factory(
            pred_coords, lambda shape: np.random.randn(*shape) + 273
        ),
        "pred_relhum150cm": variable_factory(
            pred_coords, lambda shape: np.random.rand(*shape) * 100
        ),
    }
    return predset_factory(data_vars)


@pytest.fixture
def combined_preds_eval_ds(predictions, eval_dataset) -> xr.Dataset:
    return xr.merge([predictions, eval_dataset], join="inner", compat="override")


def test_attach_derived_variables(combined_preds_eval_ds):
    conf = {
        "swbgt": [
            "mlde_notebooks.derived_variables.swbgt",
            {"temp": "temp150cm", "rh": "relhum150cm"},
        ]
    }
    actual_ds = data.attach_derived_variables(combined_preds_eval_ds, conf)

    expected_pred_dv = mlde_notebooks.derived_variables.swbgt(
        temp=combined_preds_eval_ds["pred_temp150cm"],
        rh=combined_preds_eval_ds["pred_relhum150cm"],
    )

    expected_target_dv = mlde_notebooks.derived_variables.swbgt(
        temp=combined_preds_eval_ds["target_temp150cm"],
        rh=combined_preds_eval_ds["target_relhum150cm"],
    )

    assert np.all(actual_ds["pred_swbgt"] == expected_pred_dv)

    assert np.all(actual_ds["target_swbgt"] == expected_target_dv)

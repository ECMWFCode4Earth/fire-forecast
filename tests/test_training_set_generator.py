from pathlib import Path

import pytest
import xarray as xr

from fire_forecast.data_preparation.training_set_generator import TrainingSetGenerator

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


@pytest.fixture
def dataset():
    return xr.open_dataset(FILE_EXAMPLES / "postprocessed_data_no_ident.nc")


@pytest.fixture
def trainingsetgenerator(dataset):
    return TrainingSetGenerator(dataset)


def test_select_training_set(trainingsetgenerator):
    training_set, variable_selection = trainingsetgenerator.select_training_set(
        ["t", "cvh"]
    )
    assert training_set.shape[1] == len(variable_selection)


def test_save_training_set(trainingsetgenerator, tmp_path):
    trainingsetgenerator.save_training_set(tmp_path / "test.hdf", ["t", "cvh"])
    assert (tmp_path / "test.hdf").exists()

import numpy as np
import pytest
import xarray as xr

from fire_forecast.data_preparation.data_selector import DataSelector


@pytest.fixture
def data():
    sample_data = xr.Dataset(
        data_vars=dict(
            total_frpfire=(
                ["time", "latitude", "longitude"],
                np.array(
                    [
                        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 5, 0]],
                    ]
                ),
            ),
            total_offire=(
                ["time", "latitude", "longitude"],
                np.array(
                    [
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    ]
                ),
            ),
            total_temperature=(
                ["time", "latitude", "longitude"],
                np.random.rand(3, 4, 4),
            ),
        ),
        coords=dict(
            longitude=(["longitude"], np.arange(0, 2, 0.5)),
            latitude=(["latitude"], np.arange(0, 2, 0.5)),
            time=(
                ["time"],
                np.arange(
                    "2020-01-01T00:00:00", "2020-01-04T00:00:00", dtype="datetime64[D]"
                ).astype("datetime64[ns]"),
            ),
        ),
    )
    return sample_data


@pytest.fixture
def data_selector(data) -> DataSelector:
    return DataSelector(data)


@pytest.fixture
def days_with_enough_data(data_selector: DataSelector) -> xr.DataArray:
    return data_selector._locate_days_with_enough_data(1, 1)


def test_init(data):
    data_selector = DataSelector(data)
    assert data_selector.data.equals(data)
    assert data_selector._latitude_window_size == 3
    assert data_selector._longitude_window_size == 3
    assert data_selector._latitude_grid_size == 0.5
    assert data_selector._longitude_grid_size == 0.5
    assert data_selector._time_grid_size == np.timedelta64(1, "D").astype(
        "timedelta64[ns]"
    )


def test_locate_days_with_enough_data(data_selector: DataSelector):
    days_with_enough_data = data_selector._locate_days_with_enough_data(1, 1)
    assert set(["longitude", "latitude", "dayofyear"]) == set(
        days_with_enough_data.dims
    )
    assert days_with_enough_data.dtype == bool


def test_select_sample_coordinates(
    data_selector: DataSelector, days_with_enough_data: xr.DataArray
):
    sample_coordinates = data_selector._select_sample_coordinates(days_with_enough_data)
    assert set(["longitude", "latitude", "time"]) == set(sample_coordinates.data_vars)
    assert set(["sample", "latitude_pixel", "longitude_pixel", "time_index"]) == set(
        sample_coordinates.dims
    )


def test_select_data(data_selector: DataSelector):
    selected_data = data_selector.select_data(1, 1)
    assert set(["sample", "latitude_pixel", "longitude_pixel", "time_index"]) == set(
        selected_data.dims
    )


# TODO tests for
# _expand_latitudes_to_windows,
# _expand_longitudes_to_windows,
# _expand_times_to_windows

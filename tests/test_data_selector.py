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


@pytest.fixture
def sample_coordinates(
    data_selector: DataSelector, days_with_enough_data: xr.DataArray
) -> xr.Dataset:
    return data_selector._select_sample_coordinates(days_with_enough_data)


@pytest.fixture
def filtered_data(
    data_selector: DataSelector, sample_coordinates: xr.Dataset
) -> xr.Dataset:
    return data_selector.data.sel(
        latitude=sample_coordinates.latitude,
        longitude=sample_coordinates.longitude,
        time=sample_coordinates.time,
    )


@pytest.fixture
def sample_coordinates_numpy(
    data_selector: DataSelector, days_with_enough_data: xr.DataArray
) -> np.ndarray:
    return data_selector._retrieve_coordinates_of_entries(days_with_enough_data)


@pytest.fixture
def unfiltered_sample_coordinates(
    data_selector: DataSelector, sample_coordinates_numpy: np.ndarray
) -> xr.Dataset:
    return xr.Dataset(
        data_vars=dict(
            longitude=(
                ["sample", "longitude_pixel"],
                data_selector._expand_longitudes_to_windows(
                    sample_coordinates_numpy[2]
                ),
            ),
            latitude=(
                ["sample", "latitude_pixel"],
                data_selector._expand_latitudes_to_windows(sample_coordinates_numpy[1]),
            ),
            time=(
                ["sample", "time_index"],
                data_selector._expand_times_to_windows(
                    sample_coordinates_numpy[0]
                ).astype("datetime64[ns]"),
            ),
        ),
    )


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


def test_add_metadata(
    data_selector: DataSelector,
    filtered_data: xr.Dataset,
    sample_coordinates: xr.Dataset,
):
    filtered_data_with_metadata = data_selector._add_metadata(
        filtered_data, sample_coordinates
    )
    assert "longitude" in filtered_data_with_metadata.coords
    assert "latitude" in filtered_data_with_metadata.coords
    assert "time" in filtered_data_with_metadata.coords


def test_retrieve_coordinates_of_entries(
    data_selector: DataSelector, days_with_enough_data: xr.DataArray
):
    sample_coordinates_numpy = data_selector._retrieve_coordinates_of_entries(
        days_with_enough_data
    )
    assert isinstance(sample_coordinates_numpy, np.ndarray)
    assert sample_coordinates_numpy.shape[0] == 3


# TODO tests for
# _expand_latitudes_to_windows,
def test_expand_latitudes_to_windows(
    data_selector: DataSelector, sample_coordinates_numpy
):
    latitudes = data_selector._expand_latitudes_to_windows(sample_coordinates_numpy[1])
    assert latitudes.shape[-1] == data_selector._latitude_window_size


def test_expand_longitudes_to_windows(
    data_selector: DataSelector, sample_coordinates_numpy
):
    longitudes = data_selector._expand_longitudes_to_windows(
        sample_coordinates_numpy[2]
    )
    assert longitudes.shape[-1] == data_selector._longitude_window_size


def test_expand_times_to_windows(data_selector: DataSelector, sample_coordinates_numpy):
    times = data_selector._expand_times_to_windows(sample_coordinates_numpy[0])
    assert (
        times.shape[-1] == 2
    )  # since the test data has only dayly resolution not hourly


def test_filter_boundary(
    data_selector: DataSelector, unfiltered_sample_coordinates: xr.Dataset
):
    n_samples_unfiltered = len(unfiltered_sample_coordinates.sample)
    filtered_coordinates = data_selector._filter_boundary(unfiltered_sample_coordinates)
    n_samples_filtered = len(filtered_coordinates.sample)
    assert n_samples_unfiltered > n_samples_filtered
    assert data_selector.data.longitude.min() <= filtered_coordinates.longitude.min()
    assert data_selector.data.longitude.max() >= filtered_coordinates.longitude.max()
    assert data_selector.data.latitude.min() <= filtered_coordinates.latitude.min()
    assert data_selector.data.latitude.max() >= filtered_coordinates.latitude.max()
    assert data_selector.data.time.min() <= filtered_coordinates.time.min()
    assert data_selector.data.time.max() >= filtered_coordinates.time.max()

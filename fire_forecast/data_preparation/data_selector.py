from typing import Optional

import numpy as np
import xarray as xr
from tqdm.auto import tqdm


class DataSelector:
    def __init__(
        self,
        data: xr.Dataset,
        latitude_window_size: Optional[int] = None,
        longitude_window_size: Optional[int] = None,
    ):
        """Initializes the DataSelector class.

        Args:
            data (xr.Dataset): The dataset to be used for the data selection.
            latitude_window_size (Optional[int], optional): The latitude window
                size in degrees. Defaults to None.
            longitude_window_size (Optional[int], optional): The longitude
                window size in degrees. Defaults to None.
            time_window_size (Optional[int], optional): The time window size in
                hours. Defaults to None.
        """
        self._data = data
        self._latitude_window_size = (
            latitude_window_size if latitude_window_size is not None else 3
        )
        self._longitude_window_size = (
            longitude_window_size if longitude_window_size is not None else 3
        )
        self._check_window_sizes()

        self._latitude_grid_size = self._get_latitude_grid_size()
        self._longitude_grid_size = self._get_longitude_grid_size()
        self._time_grid_size = self._get_time_grid_size()

    @property
    def data(self) -> xr.Dataset:
        return self._data

    def select_data(self, fire_threshold: int = 1, measurement_threshold: int = 1):
        """Selects pixels that have enough fire values on one day and enough
            measurements on the following day.

        Args:
            fire_threshold (int, optional): The minimum number of fire pixels
                in a window. Defaults to 1.
            measurement_threshold (int, optional): The minimum number of
                measurements in a window. Defaults to 1.

        Returns:
            xr.Dataset: The selected data.
        """
        days_with_enough_data = self._locate_days_with_enough_data(
            fire_threshold, measurement_threshold
        )
        sample_coordinates = self._select_sample_coordinates(days_with_enough_data)
        return self.data.sel(
            latitude=sample_coordinates.latitude,
            longitude=sample_coordinates.longitude,
            time=sample_coordinates.time,
        )

    def _locate_days_with_enough_data(
        self, fire_threshold: int, measurement_threshold: int
    ) -> xr.DataArray:
        """Finds days that have enough fire values on one day and enough
            measurements on the following day.

        Args:
            fire_threshold (int): The minimum number of fire pixels in a window.
            measurement_threshold (int): The minimum number of measurements in a

        Returns:
            xr.DataArray: A boolean array indicating whether a day has enough data.
        """
        fire_values_per_day: xr.DataArray = (
            (self.data.total_frpfire > 0)
            .groupby("time.dayofyear")
            .sum(dim="time")
            .compute()
        )
        measurements_on_the_following_day: xr.DataArray = (
            (self.data.total_offire > 0)
            .groupby("time.dayofyear")
            .sum(dim="time")
            .shift(dayofyear=-1, fill_value=0)
            .compute()
        )

        days_with_enough_data = (fire_values_per_day >= fire_threshold) & (
            measurements_on_the_following_day >= measurement_threshold
        )
        return days_with_enough_data

    def _select_sample_coordinates(
        self, days_with_enough_data: xr.DataArray
    ) -> xr.Dataset:
        """Gathers the coordinates of the data that fulfills the requirements.

        Args:
            days_with_enough_data (xr.DataArray): A boolean array indicating
                whether a day has enough data.

        Returns:
            xr.DataArray: The coordinates of the data that fulfills the requirements.
        """
        sample_coordinates = []

        for day_of_year in tqdm(days_with_enough_data.dayofyear.values):
            data_of_day = days_with_enough_data.sel(dayofyear=day_of_year)
            if data_of_day.sum() == 0:
                continue
            data_of_day = data_of_day.where(data_of_day, drop=True)
            for latitude in data_of_day.latitude.values:
                for longitude in data_of_day.longitude.values:
                    if np.isnan(
                        data_of_day.sel(latitude=latitude, longitude=longitude).item()
                    ):
                        continue
                    sample_coordinates.append((day_of_year, latitude, longitude))

        sample_coordinates_numpy = np.array(sample_coordinates).T
        sample_coordinates_dataset = xr.Dataset(
            data_vars=dict(
                longitude=(
                    ["sample", "longitude_pixel"],
                    self._expand_longitudes_to_windows(sample_coordinates_numpy[2]),
                ),
                latitude=(
                    ["sample", "latitude_pixel"],
                    self._expand_latitudes_to_windows(sample_coordinates_numpy[1]),
                ),
                time=(
                    ["sample", "time_index"],
                    self._expand_times_to_windows(sample_coordinates_numpy[0]).astype(
                        "datetime64[ns]"
                    ),
                ),
            ),
        )
        return self._filter_boundary(sample_coordinates_dataset)

    def _expand_latitudes_to_windows(self, latitudes: np.ndarray) -> np.ndarray:
        """Expands the latitude array to include the window size"""
        latitude_range = (
            (self._latitude_window_size - 1) // 2
        ) * self._latitude_grid_size
        shifts = np.arange(
            -latitude_range,
            latitude_range + self._latitude_grid_size,
            self._latitude_grid_size,
        )
        return latitudes[:, np.newaxis] + shifts[np.newaxis, :]

    def _expand_longitudes_to_windows(self, longitudes: np.ndarray) -> np.ndarray:
        """Expands the longitude array to include the window size"""
        longitude_range = (
            (self._longitude_window_size - 1) // 2
        ) * self._longitude_grid_size
        shifts = np.arange(
            -longitude_range,
            longitude_range + self._longitude_grid_size,
            self._longitude_grid_size,
        )
        return longitudes[:, np.newaxis] + shifts[np.newaxis, :]

    def _expand_times_to_windows(self, days_of_year: np.ndarray) -> np.ndarray:
        """Expands the time array to include the window size"""
        times = (
            self.data.time.min().values.astype("datetime64[Y]")
            + days_of_year.astype("timedelta64[D]")
            - np.timedelta64(1, "D")
        ).astype("datetime64[ns]")
        shifts = np.arange(
            np.timedelta64(0).astype(self._time_grid_size),
            np.timedelta64(48, "h").astype(self._time_grid_size),
            self._time_grid_size,
            dtype=type(self._time_grid_size),
        )
        return times[:, np.newaxis] + shifts[np.newaxis, :]

    def _filter_boundary(self, sample_coordinates: xr.Dataset) -> xr.Dataset:
        """Filters the sample coordinates to only include those that are within
        the data boundaries"""
        sample_coordinates = sample_coordinates.where(
            (
                sample_coordinates.latitude.isel(latitude_pixel=0)
                >= self.data.latitude.min()
            )
            & (
                sample_coordinates.latitude.isel(latitude_pixel=-1)
                <= self.data.latitude.max()
            )
            & (
                sample_coordinates.longitude.isel(longitude_pixel=0)
                >= self.data.longitude.min()
            )
            & (
                sample_coordinates.longitude.isel(longitude_pixel=-1)
                <= self.data.longitude.max()
            )
            & (sample_coordinates.time.isel(time_index=-1) <= self.data.time.max()),
            drop=True,
        )
        return sample_coordinates

    def _get_latitude_grid_size(self) -> float:
        """Returns the latitude grid size in degrees"""
        grid_sizes = self.data.latitude.diff("latitude")
        if np.unique(grid_sizes).shape[0] == 1:
            return grid_sizes.mean().item()
        else:
            raise ValueError("Latitude grid size is not uniform")

    def _get_longitude_grid_size(self) -> float:
        """Returns the longitude grid size in degrees"""
        grid_sizes = self.data.longitude.diff("longitude")
        if np.unique(grid_sizes).shape[0] == 1:
            return grid_sizes.mean().item()
        else:
            raise ValueError("Longitude grid size is not uniform")

    def _get_time_grid_size(self) -> np.timedelta64:
        """Returns the temporal resolution"""
        grid_sizes = self.data.time.diff("time").astype("timedelta64[ns]")
        if np.unique(grid_sizes).shape[0] == 1:
            return grid_sizes.mean().values
        else:
            raise ValueError("Time grid size is not uniform")

    def _check_window_sizes(self):
        assert (
            self._latitude_window_size - 1
        ) % 2 == 0, "Latitude window size must be odd"
        assert (
            self._longitude_window_size - 1
        ) % 2 == 0, "Longitude window size must be odd"

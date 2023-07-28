import warnings
from typing import Optional

import numpy as np
import pandas as pd
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

    def select_data_with_shift(
        self, shift=int, fire_threshold: int = 1, measurement_threshold: int = 1
    ) -> xr.Dataset:
        """Selects pixels that have enough fire values on one day and enough
            measurements on the following day. The data is shifted by the
            specified amount of hours A shift value of 1 will correspond to
            the time frame from 23:00 on one day until 22:00 on the following.

        Args:
            shift (int): The amount of hours to shift the data.
            fire_threshold (int, optional): The minimum number of fire pixels
                in a window. Defaults to 1.
            measurement_threshold (int, optional): The minimum number of
                measurements in a window. Defaults to 1.

        Returns:
            xr.Dataset: The selected data.
        """

        original_times = self.data.time.values
        shifted_times = original_times + self._time_grid_size * shift
        self._data = self.data.assign_coords(time=shifted_times)

        selected_data = self.select_data(fire_threshold, measurement_threshold)

        self._data = self.data.assign_coords(time=original_times)
        selected_data = selected_data.assign_coords(
            time=selected_data.time - self._time_grid_size * shift
        )
        return selected_data

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
        filtered_data = self.data.sel(
            latitude=sample_coordinates.latitude,
            longitude=sample_coordinates.longitude,
            time=sample_coordinates.time,
        )
        filtered_data = self._add_metadata(
            filtered_data, sample_coordinates, fire_threshold, measurement_threshold
        )
        return filtered_data

    def _add_metadata(
        self,
        filtered_data: xr.Dataset,
        sample_coordinates: xr.Dataset,
        fire_threshold: int,
        measurement_threshold: int,
    ):
        """Adds metadata to the filtered data.

        Args:
            filtered_data (xr.Dataset): The filtered data.
            sample_coordinates (xr.Dataset): The sample coordinates.
            fire_threshold (int): The minimum number of fire pixels in a window.
            measurement_threshold (int): The minimum number of measurements in a window.

        Returns:
            xr.Dataset: The filtered data with metadata.
        """
        filtered_data = filtered_data.assign_coords(
            latitude=sample_coordinates.latitude,
            longitude=sample_coordinates.longitude,
            time=sample_coordinates.time,
        )
        filtered_data = filtered_data.assign_attrs(
            dict(
                fire_threshold=fire_threshold,
                measurement_threshold=measurement_threshold,
            )
        )
        return filtered_data

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

        sample_coordinates_numpy = self._retrieve_coordinates_of_entries(
            days_with_enough_data
        )
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

    def _retrieve_coordinates_of_entries(
        self, days_with_enough_data: xr.DataArray
    ) -> np.ndarray:
        """Retrieves the coordinates of the data that fulfills the requirements.

        Args:
            days_with_enough_data (xr.DataArray): A boolean array indicating
                whether a day has enough data.

        Returns:
            np.ndarray: The coordinates of the data that fulfills the requirements.
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
        return np.array(sample_coordinates).T

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
                sample_coordinates.latitude.min("latitude_pixel")
                >= self.data.latitude.min()
            )
            & (
                sample_coordinates.latitude.max("latitude_pixel")
                <= self.data.latitude.max()
            )
            & (
                sample_coordinates.longitude.min("longitude_pixel")
                >= self.data.longitude.min()
            )
            & (
                sample_coordinates.longitude.max("longitude_pixel")
                <= self.data.longitude.max()
            )
            & (sample_coordinates.time.min("time_index") >= self.data.time.min())
            & (sample_coordinates.time.max("time_index") <= self.data.time.max()),
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


class DataCutter:
    """Class to cut data that is already devided into 3x3 pixels into samples."""

    def __init__(self, data: xr.Dataset) -> None:
        self._data = data

    @property
    def data(self) -> xr.Dataset:
        return self._data

    def cut_data_shifted(
        self,
        shift: int = 1,
        fire_threshold: int = 1,
        measurement_threshold: int = 1,
    ) -> xr.Dataset:
        """Cuts the data into samples with the specified shift.

        Args:
            shift (int, optional): The amount of hours to shift the data.
                Defaults to 1.
            fire_threshold (int, optional): The minimum number of fire pixels
                in a window. Defaults to 1.
            measurement_threshold (int, optional): The minimum number of
                measurements in a window. Defaults to 1.

        Returns:
            xr.Dataset: The cut data.
        """
        original_times = self.data.time.values
        shifted_times = original_times + np.timedelta64(shift, "h")
        self._data = self.data.assign_coords(
            time=(("sample", "time_index"), shifted_times)
        )

        cut_data = self.cut_data(
            fire_threshold=fire_threshold,
            measurement_threshold=measurement_threshold,
        )

        self._data = self.data.assign_coords(
            time=(("sample", "time_index"), original_times)
        )
        cut_data = cut_data.assign_coords(
            time=(
                ("sample", "time_index"),
                cut_data.time.values - np.timedelta64(shift, "h"),
            )
        )
        return cut_data

    def cut_data(
        self,
        fire_threshold: int = 1,
        measurement_threshold: int = 1,
    ):
        days_with_enough_data = self._get_days_with_enough_data(
            fire_threshold=fire_threshold,
            measurement_threshold=measurement_threshold,
        )
        sample_coordinates = self._extract_sample_coordinates(days_with_enough_data)
        filtered_data = self.data.sel(
            sample=sample_coordinates.sample.astype(int),
            time_index=sample_coordinates.time_index.astype(int),
        )
        filtered_data = filtered_data.rename(
            new_sample="sample", new_time_index="time_index"
        )
        return filtered_data

    def _get_days_with_enough_data(
        self,
        fire_threshold: int = 1,
        measurement_threshold: int = 1,
    ) -> xr.DataArray:
        center_pixel_data = self.data.isel(
            latitude_pixel=len(self.data.latitude_pixel) // 2,
            longitude_pixel=len(self.data.longitude_pixel) // 2,
        ).compute()
        days_with_enough_data_collection = []
        for sample in tqdm(
            center_pixel_data.sample.values, desc="Collecting days with enough data"
        ):
            center_pixel_sample = center_pixel_data.sel(sample=sample)
            center_pixel_sample = center_pixel_sample.assign_coords(
                time_index=center_pixel_sample.time.values
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fire_values_per_day: xr.DataArray = (
                    (center_pixel_sample.total_frpfire > 0)
                    .assign_coords(
                        day=center_pixel_sample.time.astype("datetime64[D]").astype(
                            "datetime64[ns]"
                        )
                    )
                    .groupby("day")
                    .sum(dim="time_index")
                )
                measurement_values_per_day: xr.DataArray = (
                    (center_pixel_sample.total_offire > 0)
                    .assign_coords(
                        day=center_pixel_sample.time.astype("datetime64[D]").astype(
                            "datetime64[ns]"
                        )
                    )
                    .groupby("day")
                    .sum(dim="time_index")
                    .shift(day=-1, fill_value=0)
                )
            days_with_enough_data = (fire_values_per_day >= fire_threshold) & (
                measurement_values_per_day >= measurement_threshold
            ).expand_dims("sample").assign_coords(sample=[sample])

            days_with_enough_data = (
                days_with_enough_data.rename(day="day_index")
                .assign_coords(
                    day=(
                        ("sample", "day_index"),
                        days_with_enough_data.day.values[None, :],
                    )
                )
                .drop("day_index")
            )
            days_with_enough_data_collection.append(days_with_enough_data)
        days_with_enough_data = xr.concat(
            days_with_enough_data_collection, dim="sample"
        ).transpose("sample", "day_index")
        return days_with_enough_data.compute()

    def _extract_sample_coordinates(
        self, days_with_enough_data: xr.DataArray
    ) -> xr.DataArray:
        sliced_selection = days_with_enough_data.day.where(
            days_with_enough_data, drop=True
        )
        new_sample_time_coordinates = []
        for sample in sliced_selection.sample.values:
            for day_index in sliced_selection.day_index.values:
                if pd.isnull(
                    sliced_selection.sel(sample=sample, day_index=day_index).item()
                ):
                    continue
                time_of_day = sliced_selection.sel(
                    sample=sample, day_index=day_index
                ).values
                sample_data = self.data.sel(sample=sample)
                try:
                    time_index_of_day = (
                        sample_data.time_index.where(
                            (sample_data.time == time_of_day).compute(), drop=True
                        )
                        .compute()
                        .item()
                    )
                except ValueError:
                    continue
                new_sample_time_coordinates.append((sample, time_index_of_day))
        new_sample_time_coordinates = np.array(new_sample_time_coordinates).T

        sample_coordinates = xr.Dataset(
            data_vars=dict(
                time_index=(
                    ["new_sample", "new_time_index"],
                    self._expand_time_indices_to_windows(
                        new_sample_time_coordinates[1]
                    ),
                ),
                sample=(
                    ["new_sample"],
                    new_sample_time_coordinates[0],
                ),
            ),
        )
        return sample_coordinates

    def _expand_time_indices_to_windows(self, time_indices: np.ndarray) -> np.ndarray:
        """Expands the time array to include the window size"""
        times_of_window = time_indices[:, None] + np.arange(48)[None, :]
        return times_of_window

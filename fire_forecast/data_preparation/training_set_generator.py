from pathlib import Path
from typing import List, Tuple

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from loguru import logger
from tqdm.auto import tqdm

CHUNK_SIZE = int(1e5)


class TrainingSetGenerator:
    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def select_training_set(
        self,
        meteo_variables: List[str],
        filter_nans: bool = True,
        fire_threshold_first_day: int = 1,
        measurement_threshold_first_day: int = 1,
        fire_threshold_second_day: int = 1,
        measurement_threshold_second_day: int = 1,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate a training set from the dataset.

        Args:
            meteo_variables (List[str]): List of meteo variables to include in the
                training set.
            filter_nans (bool, optional): Whether to filter out NaNs. Defaults to True.
            fire_threshold_first_day (int, optional): Minimum number of fires to occur in
                first day. Defaults to 1.
            measurement_threshold_first_day (int, optional): Minimum number of
                measurements to occur in first day. Defaults to 1.
            fire_threshold_second_day (int, optional): Minimum number of fires to occur in
                second day. Defaults to 1.
            measurement_threshold_second_day (int, optional): Minimum number of
                measurements to occur in second day. Defaults to 1.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the training set and the list
                of variables included in the training set.
        """
        variable_selection = ["frpfire", "offire"]
        variable_selection.extend(meteo_variables.copy())
        dataset = self.dataset[variable_selection]
        # for satiellite_index in dataset.ident.values:
        logger.info("Applying first day selection...")
        satellite_dataset = dataset.where(
            (
                dataset.frpfire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(0, 24)
                ).compute()
                > 0
            ).sum("time_index")
            >= fire_threshold_first_day,
            drop=True,
        )
        satellite_dataset = satellite_dataset.where(
            (
                satellite_dataset.offire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(0, 24)
                ).compute()
                > 0
            ).sum("time_index")
            >= measurement_threshold_first_day,
            drop=True,
        )
        logger.info("Applying second day selection...")
        satellite_dataset = satellite_dataset.where(
            (
                satellite_dataset.frpfire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(24, 48)
                ).compute()
                > 0
            ).sum("time_index")
            >= fire_threshold_second_day,
            drop=True,
        )
        satellite_dataset = satellite_dataset.where(
            (
                satellite_dataset.offire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(24, 48)
                ).compute()
                > 0
            ).sum("time_index")
            >= measurement_threshold_second_day,
            drop=True,
        )
        # filter out nan values
        if filter_nans:
            logger.info("Filtering nans...")
            # filter out nan values
            for data_var in tqdm(satellite_dataset.data_vars, desc="Filtering nans"):
                data = satellite_dataset[data_var]
                if "ident" in data.dims:
                    satellite_dataset = satellite_dataset.where(
                        (~np.isnan(data))
                        .prod(
                            [
                                "longitude_pixel",
                                "latitude_pixel",
                                "time_index",
                                "ident",
                            ]
                        )
                        .compute(),
                        drop=True,
                    )
                else:
                    satellite_dataset = satellite_dataset.where(
                        (~np.isnan(data))
                        .prod(["longitude_pixel", "latitude_pixel", "time_index"])
                        .compute(),
                        drop=True,
                    )
        logger.info("Done with filtering")
        # satellite_subsets.append(satellite_dataset)
        dataset = satellite_dataset  # xr.concat(satellite_subsets, dim="sample")
        dataarrays = []
        for data_variable in tqdm(variable_selection, desc="Selecting variables"):
            dataarrays.append(dataset[data_variable].data)
        return (
            da.stack(dataarrays, axis=0).transpose(1, 0, 2, 3, 4),
            variable_selection,
        )

    def save_training_set(
        self,
        output_path: Path,
        meteo_variables: List[str],
        filter_nans: bool = True,
        fire_threshold_first_day: int = 1,
        measurement_threshold_first_day: int = 1,
        fire_threshold_second_day: int = 1,
        measurement_threshold_second_day: int = 1,
    ) -> None:
        """
        Select training set and save it to disk.

        Args:
            output_path (Path): Path to the output file.
            meteo_variables (List[str]): List of meteo variables to include in the
                training set.
            filter_nans (bool, optional): Whether to filter out NaNs. Defaults to True.
            fire_threshold_first_day (int, optional): Minimum number of fires to occur in
                first day. Defaults to 1.
            measurement_threshold_first_day (int, optional): Minimum number of
                measurements to occur in first day. Defaults to 1.
            fire_threshold_second_day (int, optional): Minimum number of fires to occur in
                second day. Defaults to 1.
            measurement_threshold_second_day (int, optional): Minimum number of
                measurements to occur in second day. Defaults to 1.
        """
        training_set, variable_selection = self.select_training_set(
            meteo_variables,
            filter_nans,
            fire_threshold_first_day,
            measurement_threshold_first_day,
            fire_threshold_second_day,
            measurement_threshold_second_day,
        )
        logger.info(f"Opening h5 file {output_path}")
        with h5py.File(output_path, "w") as f:
            training_sample_shape = training_set.shape[1:]
            dataset = f.create_dataset(
                "training_set",
                shape=(0, *training_sample_shape),
                maxshape=(None, *training_sample_shape),
                dtype=np.float32,
            )
            logger.info("Start saving dataset chunks:")
            chunk_index = 0
            while True:
                upper_bound = min((chunk_index + 1) * CHUNK_SIZE, training_set.shape[0])
                chunk = training_set[chunk_index * CHUNK_SIZE : upper_bound]  # noqa
                if chunk.shape[0] == 0:
                    break
                logger.info(f"Saving chunk {chunk_index}. Chunk shape: {chunk.shape}")
                dataset.resize((dataset.shape[0] + chunk.shape[0], *dataset.shape[1:]))
                try:
                    dataset[-chunk.shape[0] :] = chunk.compute()  # noqa
                    logger.info("Chunk saved with dask.")
                except AttributeError:
                    dataset[-chunk.shape[0] :] = chunk  # noqa
                    logger.info("Chunk saved without dask.")

                chunk_index += 1

            f.create_dataset(
                "variable_selection", data=np.array(variable_selection, dtype="S")
            )

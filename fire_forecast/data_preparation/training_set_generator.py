from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import xarray as xr


class TrainingSetGenerator:
    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def select_training_set(self, meteo_variables: List[str]) -> Tuple[np.ndarray, str]:
        """
        Generate a training set from the dataset.

        Args:
            meteo_variables (List[str]): List of meteo variables to include in the
                training set.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the training set and the list
                of variables included in the training set.
        """
        variable_selection = ["frpfire", "offire"]
        variable_selection.extend(meteo_variables.copy())
        dataset = self.dataset[variable_selection]
        satellite_subsets = []
        for satiellite_index in dataset.ident.values:
            satellite_dataset = dataset.sel(ident=satiellite_index)
            satellite_dataset = satellite_dataset.where(
                satellite_dataset.frpfire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(0, 24)
                ).sum("time_index")
                > 0,
                drop=True,
            )

            satellite_dataset = satellite_dataset.where(
                satellite_dataset.frpfire.isel(  # satellite_dataset.offire.isel(
                    longitude_pixel=1, latitude_pixel=1, time_index=slice(24, 48)
                ).sum("time_index")
                > 0,
                drop=True,
            )
            # filter out nan values
            for data_var in satellite_dataset.data_vars:
                data = satellite_dataset[data_var]
                if "ident" in data.dims:
                    satellite_dataset = satellite_dataset.where(
                        (~np.isnan(data)).prod(
                            ["longitude_pixel", "latitude_pixel", "time_index", "ident"]
                        ),
                        drop=True,
                    )
                else:
                    satellite_dataset = satellite_dataset.where(
                        (~np.isnan(data)).prod(
                            ["longitude_pixel", "latitude_pixel", "time_index"]
                        ),
                        drop=True,
                    )

            satellite_subsets.append(satellite_dataset)
        dataset = xr.concat(satellite_subsets, dim="sample")
        dataarrays = []
        for data_variable in variable_selection:
            dataarrays.append(dataset[data_variable].values)
        return np.array(dataarrays).transpose(1, 0, 2, 3, 4), variable_selection

    def save_training_set(self, output_path: Path, meteo_variables: List[str]) -> None:
        """
        Save the training set to disk.

        Args:
            output_path (Path): Path to the output file.
            meteo_variables (List[str]): List of meteo variables to include in the
                training set.
        """
        training_set, variable_selection = self.select_training_set(meteo_variables)
        with h5py.File(output_path, "w") as f:
            f.create_dataset("training_set", data=training_set)
            f.create_dataset(
                "variable_selection", data=np.array(variable_selection, dtype="S")
            )

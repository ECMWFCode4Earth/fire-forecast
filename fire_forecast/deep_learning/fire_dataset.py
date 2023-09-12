from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
from torch.utils.data import Dataset

from fire_forecast.deep_learning.utils import (
    flatten_features,
    flatten_labels_and_weights,
)


class FireDataset(Dataset):
    """Dataset for fire forecast data."""

    def __init__(self, data_path: Union[str, Path], only_center: bool = True) -> None:
        """Initialize the dataset.

        Args:
            data_path (Union[str, Path]): Path to the data file.
            only_center (bool, optional): Whether to only use the central pixel of the
                fire features and meteo features. Defaults to True.
        """
        self.only_center = only_center
        self._data_path = Path(data_path)
        (
            self.fire_features,
            self.meteo_features,
            self.labels,
            self.data_variables,
        ) = self._read_data()
        self.n_samples = self.labels.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.fire_features[idx], self.meteo_features[idx], self.labels[idx]

    @property
    def input_size(self):
        return len(flatten_features(self.fire_features[0], self.meteo_features[0]))

    @property
    def output_size(self):
        return len(flatten_labels_and_weights(self.labels[0])[0])

    def _read_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read the data from the file and splits it into fire features, meteo features,
        labels and data variables.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
                the fire features, meteo features, labels and data variables."""
        with h5py.File(self._data_path, "r") as file:
            if self.only_center:
                data = file["training_set"][:, :, :, 1:2, 1:2]
            else:
                data = file["training_set"][:]
            data_variables = file["variable_selection"][:]

        fire_features = data[:, 0:2, :24, :, :]
        meteo_features = data[:, 2:, :, :, :]

        if self.only_center:
            labels = data[:, 0:2, 24:, 0, 0]
        else:
            labels = data[:, 0:2, 24:, 1, 1]
        return fire_features, meteo_features, labels, data_variables

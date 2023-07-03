from pathlib import Path
from typing import Union

import h5py
from torch.utils.data import Dataset

from fire_forecast.deep_learning.utils import (
    flatten_features,
    flatten_labels_and_weights,
)


class FireDataset(Dataset):
    def __init__(self, data_path: Union[str, Path]) -> None:
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

    def _read_data(self):
        with h5py.File(self._data_path, "r") as file:
            data = file["training_set"][:]
            data_variables = file["variable_selection"][:]
        labels = data[:, 0:2, 24:, 1, 1]
        fire_features = data[:, 0:2, :24, :, :]
        meteo_features = data[:, 2:, :, :, :]
        return fire_features, meteo_features, labels, data_variables

from typing import Tuple

import numpy as np


def flatten_features(
    fire_features: np.ndarray, meteo_features: np.ndarray
) -> np.ndarray:
    # for multiple data points
    if len(fire_features.shape) == 5:
        fire_features = fire_features.reshape(fire_features.shape[0], -1)
        meteo_features = meteo_features.reshape(meteo_features.shape[0], -1)
        return np.concatenate((fire_features, meteo_features), axis=1)
    # for just a single datapoint
    elif len(fire_features.shape) == 4:
        fire_features = fire_features.flatten()
        meteo_features = meteo_features.flatten()
        return np.concatenate((fire_features, meteo_features), axis=0)


def flatten_labels_and_weights(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # for multiple data points
    if len(labels.shape) == 3:
        target_values = labels[:, 0].reshape(labels.shape[0], -1)
        weights = labels[:, 1].reshape(labels.shape[0], -1)
        return target_values, weights
    # for just a single datapoint
    elif len(labels.shape) == 2:
        target_values = labels[0]
        weights = labels[1]
        return target_values, weights

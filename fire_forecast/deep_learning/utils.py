from typing import Tuple

import numpy as np
import yaml


def flatten_features(
    fire_features: np.ndarray, meteo_features: np.ndarray
) -> np.ndarray:
    """Flatten the fire and meteo features into a single array for prediction by model.

    Args:
        fire_features (np.ndarray): Fire features.
        meteo_features (np.ndarray): Meteo features.

    Returns:
        np.ndarray: Flattened and combined features.
    """
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
    """Flatten the labels and weights into a single array for prediction by model.

    Args:
        labels (np.ndarray): Labels and weights.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Flattened labels and weights.
    """
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


def read_config(config_path: str) -> dict:
    """Read the config file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Config dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

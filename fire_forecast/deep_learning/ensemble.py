import numpy as np
import torch
import torch.nn as nn

from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import flatten_features


class Ensemble:
    def __init__(self, *configs) -> None:
        """Load ensemble of models from config files.

        Args:
            *configs (str): Paths to config files.
        """
        self.models: list[nn.Module] = []
        for config in configs:
            self.models.append(load_model_from_config(config))
        [model.eval() for model in self.models]

    def predict(
        self, fire_features: np.ndarray, meteo_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the fire forecast for the given features.

        Args:
            fire_features (np.ndarray): Fire features.
            meteo_features (np.ndarray): Meteo features.

        Returns:
            tuple[np.ndarray, np.ndarray]: Mean and standard deviation of the predictions.
        """
        predictions = []
        for model in self.models:
            features = torch.from_numpy(
                flatten_features(fire_features, meteo_features)
            ).to(model.parameters().__next__().device)
            predictions.append(model(features).to("cpu"))
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

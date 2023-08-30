from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fire_forecast.deep_learning.fire_dataset import FireDataset
from fire_forecast.deep_learning.losses import load_loss_by_name
from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.utils import (
    flatten_features,
    flatten_labels_and_weights,
)


class Iterator:
    def __init__(self, config: dict):
        """Initialize the iterator.

        Args:
            config (dict): Config dictionary.
        """
        self._config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = load_loss_by_name(self._config["training"]["loss_function"])
        self.train_dataset = FireDataset(self._config["data"]["train_path"])
        self.validation_dataset = FireDataset(self._config["data"]["validation_path"])
        self.test_dataset = FireDataset(self._config["data"]["test_path"])
        self._initialize_feature_normalization()
        self.model = load_model_from_config(self._config["model"])
        self._learning_rate = self._config["training"]["learning_rate"]
        self._optimizer = self._get_optimizer_from_config(self._config["training"])
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self._config["training"]["batch_size"],
            shuffle=True,
        )
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self._config["training"]["batch_size"],
            shuffle=True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self._config["training"]["batch_size"],
            shuffle=True,
        )
        self._output_path = Path(self._config["output"]["path"])
        self._checkpoint_interval = self._config["output"]["checkpoint_interval"]

        self.epoch = 0

    def train(self):
        """Train the model."""
        self.model = self.model.to(self.device)
        self._output_path.mkdir(exist_ok=True)
        self._save_config()
        self._initialize_validation_loss_file()
        epoch_tqdm = tqdm(range(self._config["training"]["epochs"]), desc="Epochs")

        for epoch in epoch_tqdm:
            self.epoch = epoch
            batch_tqdm = tqdm(
                self.train_dataloader,
                desc="Batches",
                total=len(self.train_dataloader),
                leave=False,
            )
            for i, data in enumerate(batch_tqdm):
                self._train_batch(data, batch_tqdm)
            if self.epoch % self._checkpoint_interval == 0:
                self._save_checkpoint()
            validation_loss = self._validate(epoch_tqdm)
            self._log_validation_loss(validation_loss)
            self._update_learning_rate()
        self._save_checkpoint()

    def _train_batch(self, data: tuple, batch_tqdm: tqdm):
        """Train a batch.

        Args:
            data (tuple): Tuple of fire features, meteo features and labels.
            batch_tqdm (tqdm): Tqdm object for the batch.
        """
        self.model.train()
        fire_features, meteo_features, labels = data
        features = torch.from_numpy(flatten_features(fire_features, meteo_features)).to(
            self.device
        )
        predictions = self.model(features)
        target_values, weights = flatten_labels_and_weights(labels)
        target_values = target_values.to(self.device)
        weights = weights.to(self.device)
        loss = self.criterion(predictions, target_values, weights)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        batch_tqdm.set_postfix({"loss": loss.item()})

    def _validate(self, epoch_tqdm: tqdm):
        """Validate the model.

        Args:
            epoch_tqdm (tqdm): Tqdm object for the epoch.

        Returns:
            torch.Tensor: Validation loss.
        """
        self.model.eval()
        validation_losses = []
        with torch.no_grad():
            for data in self.validation_dataloader:
                fire_features, meteo_features, labels = data
                features = torch.from_numpy(
                    flatten_features(fire_features, meteo_features)
                ).to(self.device)
                predictions = self.model(features)
                target_values, weights = flatten_labels_and_weights(labels)
                target_values = target_values.to(self.device)
                weights = weights.to(self.device)
                validation_losses.append(
                    self.criterion(predictions, target_values, weights)
                )
        validation_loss = torch.mean(torch.stack(validation_losses))
        epoch_tqdm.set_postfix({"validation_loss": validation_loss.item()})
        self.model.train()
        return validation_loss

    def _update_learning_rate(self):
        """Update the learning rate according to the config."""
        if self._config["training"]["learning_rate_development"]["type"] == "linear":
            self._learning_rate -= self._config["training"][
                "learning_rate_development"
            ]["amount"]
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self._learning_rate
        elif (
            self._config["training"]["learning_rate_development"]["type"]
            == "exponential"
        ):
            self._learning_rate *= self._config["training"][
                "learning_rate_development"
            ]["amount"]
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self._learning_rate
        else:
            raise ValueError(
                f"Unknown learning rate development type:"
                f"{self._config['training']['learning_rate_development']['type']}"
            )

    def _initialize_feature_normalization(self):
        """Initialize the feature normalization."""
        try:
            if self._config["model"]["model_args"]["feature_normalization"]:
                try:
                    if (
                        self._config["model"]["model_args"]["mean"] is not None
                        or self._config["model"]["model_args"]["std"] is not None
                    ):
                        return
                except KeyError:
                    pass
                logger.info("Initializing feature normalization.")
                total_train_dataset = []
                for data in self.train_dataset:
                    fire_features, meteo_features, labels = data
                    total_train_dataset.append(
                        flatten_features(fire_features, meteo_features)
                    )
                total_train_dataset = np.array(total_train_dataset)
                feature_means = total_train_dataset.mean(axis=0)
                feature_stds = total_train_dataset.std(axis=0)

                self._config["model"]["model_args"]["mean"] = feature_means.tolist()
                self._config["model"]["model_args"]["std"] = feature_stds.tolist()
        except KeyError:
            pass

    def _initialize_validation_loss_file(self):
        """Initialize the validation loss file."""
        with open(self._output_path / "validation_loss.txt", "w") as file:
            file.write("epoch,validation_loss\n")

    def _log_validation_loss(self, validation_loss: torch.Tensor):
        """Log the validation loss to a file.

        Args:
            validation_loss (torch.Tensor): Validation loss.
        """
        with open(self._output_path / "validation_loss.txt", "a") as file:
            file.write(f"{self.epoch},{validation_loss.item()}\n")

    def _save_config(self):
        """Save the config file."""
        if self._config["data"]["variables"] is None:
            self._config["data"]["variables"] = (
                np.char.decode(self.train_dataset.data_variables).astype(str).tolist()
            )
        if self._config["model"]["model_args"]["final_activation"] == nn.ReLU:
            self._config["model"]["model_args"]["final_activation"] = "ReLU"
        if self._config["model"]["model_args"]["activation"] == nn.ReLU:
            self._config["model"]["model_args"]["activation"] = "ReLU"
        elif self._config["model"]["model_args"]["activation"] == nn.LeakyReLU:
            self._config["model"]["model_args"]["activation"] = "LeakyReLU"

        with open(self._output_path / "config.yaml", "w") as file:
            yaml.dump(self._config, file, default_flow_style=False, explicit_start=True)

    def _save_checkpoint(self):
        """Save a checkpoint of the model."""
        torch.save(
            self.model.state_dict(), self._output_path / f"checkpoint_{self.epoch}.pt"
        )

    def _get_optimizer_from_config(self, training_config: dict) -> optim.Optimizer:
        """Get the optimizer from the config.

        Args:
            training_config (dict): Training config.

        Returns:
            optim.Optimizer: Optimizer.
        """
        optimizer_name = training_config["optimizer_name"]
        if optimizer_name == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self._learning_rate,
                **training_config["optimizer_args"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

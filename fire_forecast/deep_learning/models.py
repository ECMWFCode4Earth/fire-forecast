from typing import List

import torch
import torch.nn as nn


def load_model_from_config(model_config: dict) -> nn.Module:
    """Load a model from a config dictionary.

    Args:
        model_config (dict): Model config dictionary.

    Returns:
        nn.Module: Model.
    """
    model = None
    model_type = model_config["name"]
    if model_type == "FullyConnectedForecaster":
        model = FullyConnectedForecaster(**model_config["model_args"])
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    if model_config["checkpoint_path"] is not None:
        model.load_state_dict(torch.load(model_config["checkpoint_path"]))
    return model


class FullyConnectedForecaster(nn.Module):
    """Simple fully connected neural network for fire forecast."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: List[int],
        activation: nn.Module = nn.ReLU,
        final_activation=None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        """Initialize the fully connected forecaster.

        Args:
            input_size (int): Number of input nodes.
            output_size (int): Number of output nodes.
            hidden_layer_sizes (List[int]): List of hidden layer sizes.
            activation (nn.Module, optional): Activation function to use.
                Defaults to nn ReLU.
            final_activation ([type], optional): Activation function after the final
                layer. Defaults to None (referrs to no activation).
            dropout (float, optional): Dropout probability. Defaults to 0.0. If dropout is
                0.0, no dropout layers are introduced.
            batch_norm (bool, optional): Whether to use batch normalization.
                Defaults to False.
        """
        super(FullyConnectedForecaster, self).__init__()
        self.layers = []
        n_input_nodes = [input_size] + list(hidden_layer_sizes)
        n_output_nodes = list(hidden_layer_sizes) + [output_size]

        for n_in, n_out in zip(n_input_nodes, n_output_nodes):
            self.layers.append(nn.Linear(n_in, n_out))
            if not n_out == output_size:
                self.layers.append(activation())
            else:
                if final_activation is not None:
                    self.layers.append(final_activation())
            if batch_norm and not n_out == output_size and not n_in == input_size:
                self.layers.append(nn.BatchNorm1d(n_out))
            if dropout > 0.0 and not n_out == output_size:
                self.layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.model(x)

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
        if model_config["model_args"]["final_activation"] == "ReLU":
            model_config["model_args"]["final_activation"] = nn.ReLU
        if model_config["model_args"]["activation"] == "ReLU":
            model_config["model_args"]["activation"] = nn.ReLU
        elif model_config["model_args"]["activation"] == "LeakyReLU":
            model_config["model_args"]["activation"] = nn.LeakyReLU
        model = FullyConnectedForecaster(**model_config["model_args"])

    elif model_type == "ResidualNetwork":
        if model_config["model_args"]["final_activation"] == "ReLU":
            model_config["model_args"]["final_activation"] = nn.ReLU
        if model_config["model_args"]["activation"] == "ReLU":
            model_config["model_args"]["activation"] = nn.ReLU
        elif model_config["model_args"]["activation"] == "LeakyReLU":
            model_config["model_args"]["activation"] = nn.LeakyReLU
        model = ResidualNetwork(**model_config["model_args"])

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
        attention: bool = False,
        num_heads: int = 2,
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
        self.attention = attention
        self.layers = []
        n_input_nodes = [input_size] + list(hidden_layer_sizes)
        n_output_nodes = list(hidden_layer_sizes) + [output_size]

        if self.attention:
            self.attention_layer = nn.MultiheadAttention(input_size, num_heads)
            n_input_nodes[0] = 2 * input_size
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
        if self.attention:
            original_shape = x.shape
            if len(original_shape) == 1:
                x = x.unsqueeze(0)
            y, _ = self.attention_layer(x, x, x)
            x = torch.concat((x, y), dim=1)
            if len(original_shape) == 1:
                x = x.squeeze(0)
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        internal_layers: list[int],
        activation: nn.Module = nn.ReLU,
    ):
        super(ResidualBlock, self).__init__()
        self.layers = []

        n_input_nodes = [input_size] + list(internal_layers)
        n_output_nodes = list(internal_layers) + [input_size]
        for i, (n_in, n_out) in enumerate(zip(n_input_nodes, n_output_nodes)):
            self.layers.append(nn.Linear(n_in, n_out))
            if not i == len(n_input_nodes) - 1:
                self.layers.append(activation())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x


class ResidualNetwork(nn.Module):
    """Residual network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: list[list[int]],
        activation: nn.Module = nn.ReLU,
        final_activation=None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        attention: bool = False,
        num_heads: int = 2,
    ):
        super(ResidualNetwork, self).__init__()
        self.layers = []
        self.attention = attention

        n_input_nodes = [input_size] + list(hidden_layer_sizes)
        n_output_nodes = list(hidden_layer_sizes) + [output_size]

        if self.attention:
            self.attention_layer = nn.MultiheadAttention(input_size, num_heads)
            # n_input_nodes[0] = 2*input_size

        for n_in, n_out in zip(n_input_nodes, n_output_nodes):
            if isinstance(n_in, list):
                n_in = n_in[-1]
            if isinstance(n_out, list):
                self.layers.append(ResidualBlock(n_in, n_out, activation=activation))
            else:
                self.layers.append(nn.Linear(n_in, n_out))

            if not n_out == output_size:
                self.layers.append(activation())
            else:
                if final_activation is not None:
                    self.layers.append(final_activation())
            if batch_norm and not n_out == output_size and not n_in == input_size:
                self.layers.append(
                    nn.BatchNorm1d(n_out if isinstance(n_out, int) else n_out[-1])
                )
            if dropout > 0.0 and not n_out == output_size:
                self.layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention:
            original_shape = x.shape
            if len(original_shape) == 1:
                x = x.unsqueeze(0)
            y, _ = self.attention_layer(x, x, x)
            # x = torch.concat((x, y), dim=1)
            x = x * y
            if len(original_shape) == 1:
                x = x.squeeze(0)
        return self.model(x)

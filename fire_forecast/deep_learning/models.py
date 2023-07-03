import torch.nn as nn


class FullyConnectedForecaster(nn.Module):
    def __init__(self, input_size, output_size, *hidden_layer_sizes):
        super(FullyConnectedForecaster, self).__init__()
        self.layers = []
        n_input_nodes = [input_size] + list(hidden_layer_sizes)
        n_output_nodes = list(hidden_layer_sizes) + [output_size]

        for n_in, n_out in zip(n_input_nodes, n_output_nodes):
            self.layers.append(nn.Linear(n_in, n_out))
            self.layers.append(nn.ReLU())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

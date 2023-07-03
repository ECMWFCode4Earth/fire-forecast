import torch
import torch.nn as nn


class WeightedMAELoss(nn.Module):
    def __init__(self):
        super(WeightedMAELoss, self).__init__()

    def forward(self, predictions, labels, weights):
        loss = torch.abs((predictions - labels) * weights**2).sum()
        # devide by number of non-zero-weights
        loss = loss / (weights != 0).sum()
        return loss

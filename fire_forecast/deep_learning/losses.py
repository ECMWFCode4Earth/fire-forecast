import torch
import torch.nn as nn


class WeightedMAELoss(nn.Module):
    """Loss function for weighted MAE loss.

    Each diffence between prediction and label is weighted by the weight of the
        corresponding label before an average is taken over the values with non-zero
        weights.
    """

    def __init__(self):
        super(WeightedMAELoss, self).__init__()

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            predictions (torch.Tensor): Predictions of the model.
            labels (torch.Tensor): Labels of the data.
            weights (torch.Tensor): Weights of the labels.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = torch.abs((predictions - labels) * weights**2)
        # devide by number of non-zero-weights
        loss = loss / (weights != 0).sum()
        return loss


class WeightedL1Loss(nn.Module):
    """Loss function for weighted MAE loss.

    Each diffence between prediction and label is weighted by the weight of the
        corresponding label before everything is summed up.
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the loss function.

        Args:
            predictions (torch.Tensor): Predictions of the model.
            labels (torch.Tensor): Labels of the data.
            weights (torch.Tensor): Weights of the labels.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = torch.abs((predictions - labels) * weights**2).sum()
        return loss

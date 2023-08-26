import torch
import torch.nn as nn


def load_loss_by_name(loss_name: str) -> nn.Module:
    """Load a loss function by name.

    Args:
        loss_name (str): Name of the loss function.

    Raises:
        ValueError: If the loss function is not supported.

    Returns:
        nn.Module: Loss function.
    """
    if loss_name == "WeightedMAELoss":
        return WeightedMAELoss()
    elif loss_name == "WeightedL1Loss":
        return WeightedL1Loss()
    elif loss_name == "WeightedL1RootLoss":
        return WeightedL1RootLoss()
    elif loss_name == "L1Loss":
        return L1Loss()
    else:
        raise ValueError(f"Loss function {loss_name} not supported.")


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
        loss = torch.abs((predictions - labels) * weights)
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
        loss = torch.abs((predictions - labels) * weights).sum()
        return loss


class WeightedL1RootLoss(nn.Module):
    """Loss function for weighted MAE loss.

    Each diffence between prediction and label is weighted by the weight of the
        corresponding label before everything is summed up.
    """

    def __init__(self):
        super(WeightedL1RootLoss, self).__init__()

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
        loss = torch.abs((predictions**0.1 - labels**0.1) * weights).sum()
        return loss


class L1Loss(nn.Module):
    """Loss function for weighted MAE loss (weight is ignored)."""

    def __init__(self):
        super(L1Loss, self).__init__()

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
        loss = torch.abs((predictions - labels)).sum()
        return loss

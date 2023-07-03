import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fire_forecast.deep_learning.fire_dataset import FireDataset
from fire_forecast.deep_learning.losses import WeightedMAELoss
from fire_forecast.deep_learning.models import FullyConnectedForecaster
from fire_forecast.deep_learning.utils import (
    flatten_features,
    flatten_labels_and_weights,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network to predict fire development."
    )
    parser.add_argument("data_path", type=str, help="Path to the training set.")
    parser.add_argument("output_path", type=str, help="Path to the output file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument(
        "--hidden_layers",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="Number of neurons in each hidden layer.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size.")
    parser.add_argument(
        "--validation_size", type=float, default=0.2, help="Validation set size."
    )

    return parser.parse_args()


def main():
    args = get_args()
    data_path = Path(args.data_path)
    # output_path = Path(args.output_path)
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    hidden_layers = args.hidden_layers
    learning_rate = args.learning_rate

    dataset = FireDataset(data_path)

    test_size = int(args.test_size * len(dataset))
    validation_size = int(args.validation_size * len(dataset))
    train_size = len(dataset) - test_size - validation_size

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size, test_size]
    )

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = FullyConnectedForecaster(
        dataset.input_size, dataset.output_size, *hidden_layers
    )

    criterion = WeightedMAELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader, 0), desc="Batches"):
            fire_features, meteo_features, labels = data
            optimizer.zero_grad()
            features = flatten_features(fire_features, meteo_features)
            outputs = model(torch.from_numpy(features))
            target_values, weights = flatten_labels_and_weights(labels)
            loss = criterion(outputs, target_values, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0


if __name__ == "__main__":
    main()

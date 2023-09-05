import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from fire_forecast.data_preparation.data_selector import DataCutter


def get_args():
    parser = argparse.ArgumentParser(description="Script to select data from dataset.")
    # argument to for n filenames:
    parser.add_argument(
        "data_paths", type=str, nargs="+", help="Paths of data to read."
    )
    parser.add_argument("--output_path", type=str, help="Path to output dataset.")
    parser.add_argument(
        "--fire_number_threshold", type=int, default=1, help="Fire threshold."
    )
    parser.add_argument(
        "--measurement_threshold", type=int, default=1, help="Measurement threshold."
    )
    parser.add_argument(
        "--validation_split", type=float, default=0, help="Validation split."
    )
    parser.add_argument("--test_split", type=float, default=0, help="Test split.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    output_path = Path(args.output_path)
    full_dataset = xr.open_mfdataset(args.data_paths).compute()

    full_dataset["total_frpfire"] = full_dataset.frpfire
    full_dataset["total_offire"] = full_dataset.offire

    if args.validation_split == 0 and args.test_split == 0:
        datacutter = DataCutter(full_dataset)
        for i in range(24):
            datacutter.cut_data_shifted(
                i, args.fire_number_threshold, args.measurement_threshold
            ).to_netcdf(output_path.parent / f"{output_path.stem}_{i}.nc")

    else:
        dataset_size = len(full_dataset.sample)
        validation_size = int(dataset_size * args.validation_split)
        test_size = int(dataset_size * args.test_split)
        train_size = dataset_size - validation_size - test_size
        random_indices = np.random.permutation(dataset_size)
        train_indices = random_indices[:train_size]
        validation_indices = random_indices[
            train_size : train_size + validation_size  # noqa
        ]
        test_indices = random_indices[train_size + validation_size :]  # noqa

        train_dataset = full_dataset.isel(sample=train_indices)
        validation_dataset = full_dataset.isel(sample=validation_indices)
        test_dataset = full_dataset.isel(sample=test_indices)

        train_datacutter = DataCutter(train_dataset)
        validation_datacutter = DataCutter(validation_dataset)
        test_datacutter = DataCutter(test_dataset)

        for i in range(24):
            train_datacutter.cut_data_shifted(
                i, args.fire_number_threshold, args.measurement_threshold
            ).to_netcdf(output_path.parent / f"{output_path.stem}_train_{i}.nc")
            validation_datacutter.cut_data_shifted(
                i, args.fire_number_threshold, args.measurement_threshold
            ).to_netcdf(output_path.parent / f"{output_path.stem}_validation_{i}.nc")
            test_datacutter.cut_data_shifted(
                i, args.fire_number_threshold, args.measurement_threshold
            ).to_netcdf(output_path.parent / f"{output_path.stem}_test_{i}.nc")

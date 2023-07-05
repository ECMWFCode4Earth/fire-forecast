import argparse
from pathlib import Path

import xarray as xr
from loguru import logger

from fire_forecast.data_preparation.training_set_generator import TrainingSetGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a training set from the postprocessed data."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the postprocessed data file or directory of files that"
        "are concatenated.",
    )
    parser.add_argument("output_file", type=str, help="Path to the output file.")
    parser.add_argument(
        "meteo_variables",
        type=str,
        nargs="+",
        help="List of meteo variables to include in the training set.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_path = Path(args.input_path)
    logger.info(f"Loading input from {input_path}")
    if input_path.is_dir():
        dataset = xr.open_mfdataset(
            input_path.glob("*.nc"), concat_dim="sample", combine="nested"
        ).compute()
    elif input_path.is_file():
        dataset = xr.open_dataset(input_path)
    else:
        raise ValueError("Input path must be a directory or a file.")
    trainingsetgenerator = TrainingSetGenerator(dataset)
    logger.info(f"Saving training set to {args.output_file}")
    trainingsetgenerator.save_training_set(Path(args.output_file), args.meteo_variables)


if __name__ == "__main__":
    main()

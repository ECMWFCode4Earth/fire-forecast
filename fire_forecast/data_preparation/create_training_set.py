import argparse
from pathlib import Path

import xarray as xr
from loguru import logger

from fire_forecast.data_preparation.training_set_generator import TrainingSetGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a training set in h5 format from the postprocessed data "
        "snippets in netCDF format. Additional filtering can be applied to the data."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the postprocessed data file or directory of files that"
        "are to be concatenated.",
    )
    parser.add_argument("output_file", type=str, help="Path to the output file.")
    parser.add_argument(
        "meteo_variables",
        type=str,
        nargs="+",
        help="List of meteo variables to include in the training set.",
    )
    parser.add_argument(
        "--filename_start",
        type=str,
        default="",
        help="File name start of all files to choose from the input directory.",
    )
    parser.add_argument(
        "--filter_nans",
        type=bool,
        default=True,
        help="Whether to filter out nan values. Default: True.",
    )
    parser.add_argument(
        "--fire_number_threshold_first_day",
        type=int,
        default=1,
        help="Data with less fire occurencies (non-zero values in frpfire) on the first "
        "day are filtered out. Default: 1.",
    )
    parser.add_argument(
        "--measurement_threshold_first_day",
        type=int,
        default=1,
        help="Data with less measurements (non-zero values in offire) on the first day "
        "are filtered out. Default: 1.",
    )
    parser.add_argument(
        "--fire_number_threshold_second_day",
        type=int,
        default=1,
        help="Data with less fire occurencies (non-zero values in frpfire) on the second "
        "day are filtered out. Default: 1.",
    )
    parser.add_argument(
        "--measurement_threshold_second_day",
        type=int,
        default=1,
        help="Data with less measurements (non-zero values in offire) on the second day "
        "are filtered out. Default: 1.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_path = Path(args.input_path)
    logger.info(f"Loading input from {input_path}")
    if input_path.is_dir():
        dataset = xr.open_mfdataset(
            input_path.glob(f"{args.filename_start}*.nc"),
            concat_dim="sample",
            combine="nested",
            chunks="auto",
        )  # .compute()
    elif input_path.is_file():
        dataset = xr.open_dataset(input_path)
    else:
        raise ValueError("Input path must be a directory or a file.")
    trainingsetgenerator = TrainingSetGenerator(dataset)
    logger.info(f"Saving training set to {args.output_file}")
    trainingsetgenerator.save_training_set(
        Path(args.output_file),
        args.meteo_variables,
        args.filter_nans,
        args.fire_number_threshold_first_day,
        args.measurement_threshold_first_day,
        args.fire_number_threshold_second_day,
        args.measurement_threshold_second_day,
    )


if __name__ == "__main__":
    main()

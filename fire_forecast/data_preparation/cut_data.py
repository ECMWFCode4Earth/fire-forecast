import argparse

import xarray as xr
from loguru import logger

from fire_forecast.data_preparation.data_selector import DataCutter


def get_args():
    parser = argparse.ArgumentParser(
        description="Script to select 48 hour snippets from long timeseries."
    )
    # argument to for n filenames:
    parser.add_argument(
        "data_paths", type=str, nargs="+", help="Paths of data to read."
    )
    parser.add_argument("--output_path", type=str, help="Path to output dataset.")
    parser.add_argument(
        "--fire_number_threshold",
        type=int,
        default=1,
        help="Threshold value for number of fires in the 48 hour snippets (no threshold "
        "for intensity). Default: 1.",
    )
    parser.add_argument(
        "--measurement_threshold",
        type=int,
        default=1,
        help="Threshold value for the number of measured values in the 48 hour snippets "
        "(non-zero values in offire). Default: 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logger.info(f"Loading data from {args.data_paths}...")
    full_dataset = xr.open_mfdataset(args.data_paths).compute()
    full_dataset["total_frpfire"] = full_dataset.frpfire
    full_dataset["total_offire"] = full_dataset.offire

    dataselector = DataCutter(full_dataset)
    logger.info("Cutting data...")
    cut_data = dataselector.cut_data(
        args.fire_number_threshold, args.measurement_threshold
    )
    logger.info(f"Saving data to {args.output_path}...")
    cut_data.to_netcdf(args.output_path)
    logger.info("Done.")

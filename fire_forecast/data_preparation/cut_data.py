import argparse

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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    full_dataset = xr.open_mfdataset(args.data_paths).compute()

    full_dataset["total_frpfire"] = full_dataset.frpfire 
    full_dataset["total_offire"] = full_dataset.offire 

    dataselector = DataCutter(full_dataset)
    dataselector.cut_data(
        args.fire_number_threshold, args.measurement_threshold
    ).to_netcdf(args.output_path)

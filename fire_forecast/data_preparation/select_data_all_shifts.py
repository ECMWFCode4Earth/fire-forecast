import argparse
from pathlib import Path

import xarray as xr

from fire_forecast.data_preparation.data_selector import DataSelector


def get_args():
    parser = argparse.ArgumentParser(description="Script to select data from dataset.")
    parser.add_argument("era5_path", type=str, help="Path to ERA5 dataset.")
    parser.add_argument("era5_sfc_path", type=str, help="Path to fire dataset.")
    parser.add_argument("gfas_path", type=str, help="Path to output dataset.")
    parser.add_argument("output_path", type=str, help="Path to output dataset.")
    parser.add_argument(
        "--latitude_window_size", type=int, default=3, help="Latitude window size."
    )
    parser.add_argument(
        "--longitude_window_size", type=int, default=3, help="Longitude window size."
    )
    parser.add_argument(
        "--fire_number_threshold", type=int, default=1, help="Fire threshold."
    )
    parser.add_argument(
        "--measurement_threshold", type=int, default=1, help="Measurement threshold."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    output_path = Path(args.output_path)
    full_dataset = xr.open_mfdataset(
        [args.era5_path, args.era5_sfc_path, args.gfas_path]
    )

    full_dataset["total_frpfire"] = full_dataset.frpfire
    full_dataset["total_offire"] = full_dataset.offire

    dataselector = DataSelector(
        full_dataset, args.latitude_window_size, args.longitude_window_size
    )
    for i in range(0, 24):
        dataselector.select_data_with_shift(
            i, args.fire_number_threshold, args.measurement_threshold
        ).to_netcdf(output_path.parent / f"{output_path.stem}_{i}.nc")

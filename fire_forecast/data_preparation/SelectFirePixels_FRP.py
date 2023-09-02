import argparse
import os

import numpy as np
import xarray as xr


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to preprocess GFAS and Meteo Data for machine learning, run "
        "script e.g. with python SelectFirePixels_FRP.py <outputpath> --i "
        "<inputdatapath1> <inputdatapath2> <inputdatapath3>"
    )
    parser.add_argument("OutputPath", type=str, help="Enter FilePath to save output")
    parser.add_argument(
        "--i",
        "-InputPathList",
        nargs="+",
        default=[],
        help="Enter filepaths to inputdata (.nc fomat), needs at least a GFAS dataset "
        "with frpfire and offire variable",
    )
    args = parser.parse_args()
    return args


# SETTINGS:
args = parse_arguments()

outputpath = args.OutputPath
pathlist = args.i

"""
outputpath = "/data/input/0p5deg/ConInterp_NewData/FRP_2023_7_AllParam.nc"

pathlist = [
        "/data/input/0p5deg/ConInterp_NewData/fire_gfas_0p5_20230701_20230731.nc",
        "/data/input/0p5deg/ConInterp_NewData/meteo_pl_0p5_20230701_20230731.nc",
        "/data/input/0p5deg/ConInterp_NewData/meteo_sfc_0p5_20230701_20230731.nc",
        ]
"""

grid_size = 0.5
latitude_sample_size = 3  # neigboring lat pixels are selected too
longitude_sample_size = 3  # neigboring lon pixels are selected too
latitude_range = ((latitude_sample_size - 1) // 2) * grid_size
longitude_range = ((longitude_sample_size - 1) // 2) * grid_size

if os.path.isfile(outputpath.split(".")[0] + "SampleCoords.nc"):
    sample_coordinates_dataset = xr.open_dataset(
        outputpath.split(".")[0] + "SampleCoords.nc"
    )
else:
    DS = xr.open_mfdataset(pathlist, chunks={"latitude": 4, "longitude": 36})

    # select data with at least one measured fire of one of the satellites
    DS = DS.assign(total_frpfire=DS.frpfire)
    DS_fire = (DS.total_frpfire.sum("time") > 0).compute()

    # get lat, lon with total_frpfire > 0
    print("get coordinates")
    sample_coordinates = []
    for lat in DS_fire.latitude.values:
        for lon in DS_fire.longitude.values:
            if not DS_fire.sel(latitude=lat, longitude=lon).item():
                continue
            sample_coordinates.append((lat, lon))
    sample_coordinates = np.array(sample_coordinates).T

    # prepare new file structure
    print("prepare new file structure")
    sample_latitudes_shifts = np.arange(
        -latitude_range, latitude_range + grid_size, grid_size
    )  # start, end , step
    sample_latitudes = (
        sample_coordinates[0][:, np.newaxis] + sample_latitudes_shifts[np.newaxis, :]
    )
    sample_longitudes_shifts = np.arange(
        -longitude_range, longitude_range + grid_size, grid_size
    )  # start, end , step
    sample_longitudes = (
        sample_coordinates[1][:, np.newaxis] + sample_longitudes_shifts[np.newaxis, :]
    )
    sample_times = np.repeat([DS.time.values], len(sample_coordinates[1]), axis=0)  #
    sample_coordinates_dataset = xr.Dataset(
        data_vars=dict(
            longitude=(
                ["sample", "longitude_pixel"],
                sample_longitudes,
            ),
            latitude=(
                ["sample", "latitude_pixel"],
                sample_latitudes,
            ),
            time=(
                ["sample", "time_index"],
                sample_times.astype("datetime64[ns]"),
            ),
        ),
    )

    print("prepare coordinates in datset")
    sample_coordinates_dataset = sample_coordinates_dataset.where(
        (sample_coordinates_dataset.latitude.min("latitude_pixel") >= DS.latitude.min())
        & (
            sample_coordinates_dataset.latitude.max("latitude_pixel")
            <= DS.latitude.max()
        )
        & (
            sample_coordinates_dataset.longitude.min("longitude_pixel")
            >= DS.longitude.min()
        )
        & (
            sample_coordinates_dataset.longitude.max("longitude_pixel")
            <= DS.longitude.max()
        )
        & (sample_coordinates_dataset.time.min("time_index") >= DS.time.min())
        & (sample_coordinates_dataset.time.max("time_index") <= DS.time.max()),
        drop=True,
    )

    sample_coordinates_dataset.to_netcdf(outputpath.split(".")[0] + "SampleCoords.nc")
    del DS

DS = xr.open_mfdataset(pathlist, chunks={"latitude": 4, "longitude": 36})

print("filter data")
filtered_data = DS.sel(
    latitude=sample_coordinates_dataset.latitude,
    longitude=sample_coordinates_dataset.longitude,
)
filtered_data = filtered_data.rename_dims({"time": "time_index"})

print("assign coordinates")
filtered_data = filtered_data.assign_coords(
    latitude=sample_coordinates_dataset.latitude,
    longitude=sample_coordinates_dataset.longitude,
    time=sample_coordinates_dataset.time,
)

print("save output")
filtered_data.to_netcdf(outputpath)

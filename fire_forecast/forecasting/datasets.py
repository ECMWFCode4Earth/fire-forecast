import numpy as np
import xarray as xr
import pandas as pd

from fire_forecast import fire_model as fm


class ToyDataset:
    def __init__(self, t: int, seed: int = 0) -> None:
        """
        Toy data loader for testing purposes.

        The data loader uses the `create_ts_` functions from the `fire_model` to generate data for learning algorithms.

        Parameters
        ----------
        t : int
            Number of time steps to generate.
        seed : int
            Seed for the random number generator.
        """
        self.t = t
        self.seed = seed
        # Init rng
        self.rng = np.random.default_rng(seed=self.seed)

        # Todo: add biomass as input
        self.biomass = 100

    def get_data(self, n: int) -> xr.Dataset:
        """
        Generate data for learning algorithms.

        Description of the data:
        - `temperature`: Temperature time series.
        - `humidity`: Humidity time series.
        - `fire`: Fire time series.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        xr.Dataset
            Dataset containing the generated data.
        """
        ds = xr.Dataset(
            coords={"time": np.arange(self.t), "n": np.arange(n)},
        )
        # Add temperature
        ds["temperature"] = xr.DataArray(
            data=[fm.create_temperature_timeseries(self.t, self.rng) for _ in range(n)],
            dims=["n", "time"],
        )
        # Add humidity
        ds["humidity"] = xr.DataArray(
            data=[fm.create_humidity_timeseries(self.t, self.rng) for _ in range(n)],
            dims=["n", "time"],
        )
        # Add fire
        ds["fire"] = xr.DataArray(
            data=[
                fm.create_timeseries_fire(
                    timeseries_length=self.t,
                    biomass=self.biomass,
                    temperature_timeseries=ds["temperature"][i],
                    humidity_timeseries=ds["humidity"][i],
                    rng=self.rng,
                )
                for i in range(n)
            ],
            dims=["n", "time"],
        )

        return ds


class DataLoader:
    def __init__(self, dataset: xr.Dataset, feature_windows, target_windows) -> None:
        """
        Data loader for learning algorithms.

        The data loader transforms time series datasets using a window approach. For each feature in the dataset, a window must be specified by a starting point and an endpoint. The window is then applied to the time series. The same is done for the target variables. The resulting data is then used for training and testing.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the data with dimensions `time` and `n`.
        feature_windows : dict
            Dictionary containing the windows for the features. The keys are the feature names and the values are tuples containing the starting and ending point of the window.
        target_windows : dict
            Dictionary containing the windows for the targets. The keys are the target names and the values are tuples containing the starting and ending point of the window.
        """
        self.ds = dataset

        # Check if feature windows are valid
        for feature in feature_windows:
            if feature not in self.ds:
                raise ValueError(f"Feature '{feature}' not in dataset.")
            if feature_windows[feature][0] >= feature_windows[feature][1]:
                raise ValueError(
                    f"Invalid window for feature '{feature}': {feature_windows[feature]}"
                )
        # Check if target windows are valid
        for target in target_windows:
            if target not in self.ds:
                raise ValueError(f"Target '{target}' not in dataset.")
            if target_windows[target][0] >= target_windows[target][1]:
                raise ValueError(
                    f"Invalid window for target '{target}': {target_windows[target]}"
                )

        # Check if feature and target windows overlap
        for feature in feature_windows:
            if feature in target_windows:
                if target_windows[feature][0] < feature_windows[feature][1]:
                    raise ValueError(
                        f"Feature and target windows overlap for '{feature}': {feature_windows[feature]} and {target_windows[feature]}"
                    )

        self.feature_windows = feature_windows
        self.target_windows = target_windows

    def get_data(self) -> xr.Dataset:
        # Timesteps for features from the min and max from the windows
        feat_min = np.min(
            [self.feature_windows[feature][0] for feature in self.feature_windows]
        )
        feat_max = np.max(
            [self.feature_windows[feature][1] for feature in self.feature_windows]
        )

        # Timesteps for targets from the min and max from the windows
        targ_min = np.min(
            [self.target_windows[target][0] for target in self.target_windows]
        )
        targ_max = np.max(
            [self.target_windows[target][1] for target in self.target_windows]
        )

        min = np.min([feat_min, targ_min])
        max = np.max([feat_max, targ_max])

        timestep = np.arange(0, max - min)

        # Number of samples
        time = self.ds["time"].values
        num_samples = len(time) - len(timestep) + 1

        # Sum of all features
        n_x = 0
        for feature in self.feature_windows:
            n_x += self.feature_windows[feature][1] - self.feature_windows[feature][0]

        # Max length of feature name
        max_feature_name_length = np.max(
            [len(feature) for feature in self.feature_windows]
        )

        # Max length of target name
        max_target_name_length = np.max([len(target) for target in self.target_windows])

        # Sum of all targets
        n_y = 0
        for target in self.target_windows:
            n_y += self.target_windows[target][1] - self.target_windows[target][0]

        # Create dataset
        ds = xr.Dataset(
            coords={
                "timestep": timestep,
                "m": np.arange(num_samples),
                "n": np.arange(len(self.ds["n"])),
                "n_x": np.arange(n_x),
                "n_y": np.arange(n_y),
            },
        )

        # Add the start time for timestep to the dataset
        ds["timesteps_start"] = xr.DataArray(
            data=[self.ds["time"][i] for i in ds["m"]],
            dims=["m"],
        )

        # Add features
        for feature in self.feature_windows:
            feature_name = "f_" + feature
            ds[feature_name] = xr.DataArray(
                dims=["timestep", "m", "n"],
                coords={"timestep": ds["timestep"], "m": ds["m"], "n": ds["n"]},
            )
            start, end = self.feature_windows[feature]
            for t in range(start, end):
                # Add feature to dataset
                # The features are added for each timestep
                ds[feature_name][{"timestep": t}] = self.ds[feature][
                    {"time": slice(t, num_samples + t)}
                ].transpose("time", "n")

        # Add targets
        for target in self.target_windows:
            target_name = "t_" + target
            ds[target_name] = xr.DataArray(
                dims=["timestep", "m", "n"],
                coords={"timestep": ds["timestep"], "m": ds["m"], "n": ds["n"]},
            )
            start, end = self.target_windows[target]
            for t in range(start, end):
                # Add target to dataset
                # The targets are added for each timestep
                ds[target_name][{"timestep": t}] = self.ds[target][
                    {"time": slice(t, num_samples + t)}
                ].transpose("time", "n")

        # Create new variables X and y
        ds["X"] = xr.DataArray(
            dims=["n_x", "m", "n"],
            coords={"n_x": ds["n_x"], "m": ds["m"], "n": ds["n"]},
        )
        ds_X_feature = xr.DataArray(
            " " * max_feature_name_length,
            dims=["n_x"],
            coords={"n_x": ds["n_x"]},
        )
        ds_X_timestep = xr.DataArray(
            dims=["n_x"],
            coords={"n_x": ds["n_x"]},
        )
        n_x_start = 0
        for feature in self.feature_windows:
            feature_name = "f_" + feature
            start, end = self.feature_windows[feature]
            n_x_end = n_x_start + end - start
            ds["X"][{"n_x": slice(n_x_start, n_x_end)}] = ds[feature_name][
                {"timestep": slice(start, end)}
            ]
            ds_X_feature[{"n_x": slice(n_x_start, n_x_end)}] = feature
            ds_X_timestep[{"n_x": slice(n_x_start, n_x_end)}] = np.arange(start, end)
            n_x_start = n_x_end

        # Combine X_feature and X_timestep to multiindex and assign as coords to n_x
        multiindex = pd.MultiIndex.from_arrays(
            [ds_X_feature.values, ds_X_timestep.values],
            names=("feature", "feature_timestep"),
        )
        ds = ds.assign_coords({"n_x": multiindex})

        ds["y"] = xr.DataArray(
            dims=["n_y", "m", "n"],
            coords={"n_y": ds["n_y"], "m": ds["m"], "n": ds["n"]},
        )
        ds_y_target = xr.DataArray(
            " " * max_target_name_length,
            dims=["n_y"],
            coords={"n_y": ds["n_y"]},
        )
        ds_y_timestep = xr.DataArray(
            dims=["n_y"],
            coords={"n_y": ds["n_y"]},
        )

        n_y_start = 0
        for target in self.target_windows:
            target_name = "t_" + target
            start, end = self.target_windows[target]
            n_y_end = n_y_start + end - start
            ds["y"][{"n_y": slice(n_y_start, n_y_end)}] = ds[target_name][
                {"timestep": slice(start, end)}
            ]
            ds_y_target[{"n_y": slice(n_y_start, n_y_end)}] = target
            ds_y_timestep[{"n_y": slice(n_y_start, n_y_end)}] = np.arange(start, end)
            n_y_start = n_y_end

        # Combine y_target and y_timestep to multiindex and assign as coords to n_y
        multiindex = pd.MultiIndex.from_arrays(
            [ds_y_target.values, ds_y_timestep.values],
            names=("target", "target_timestep"),
        )
        ds.coords["n_y"] = multiindex

        # Stack m and n dimensions
        ds = ds.stack(sample=("n", "m"))
        return ds

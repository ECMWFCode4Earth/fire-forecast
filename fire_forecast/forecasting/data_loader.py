import numpy as np
import xarray as xr

from fire_forecast import fire_model as fm


class ToyLoader:
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
            data=[fm.create_ts_temperature(self.t, self.rng) for _ in range(n)],
            dims=["n", "time"],
        )
        # Add humidity
        ds["humidity"] = xr.DataArray(
            data=[fm.create_ts_humidity(self.t, self.rng) for _ in range(n)],
            dims=["n", "time"],
        )
        # Add fire
        ds["fire"] = xr.DataArray(
            data=[
                fm.create_ts_fire(
                    t=self.t,
                    biomass=self.biomass,
                    ts_temp=ds["temperature"][i],
                    ts_hum=ds["humidity"][i],
                    rng=self.rng,
                )
                for i in range(n)
            ],
            dims=["n", "time"],
        )

        return ds

import numpy as np


from fire_forecast import fire_model as fm


def test():
    """
    Test that the fire time series is created correctly.
    """
    n_tests = 100
    timeseries_length = 1000
    rng = np.random.default_rng(seed=0)
    for i in range(n_tests):
        temperature_timeseries = fm.create_temperature_timeseries(
            timeseries_length, rng, seed=i
        )
        humidity_timeseries = fm.create_humidity_timeseries(
            timeseries_length, rng, seed=i
        )
        fm.create_timeseries_fire(timeseries_length, rng=rng, seed=i)
        fm.create_timeseries_fire(timeseries_length, rng=rng, seed=i, biomass=100)
        fm.create_timeseries_fire(
            timeseries_length,
            rng=rng,
            seed=i,
            biomass=100,
            temperature_timeseries=temperature_timeseries,
        )
        fm.create_timeseries_fire(
            timeseries_length,
            rng=rng,
            seed=i,
            biomass=100,
            temperature_timeseries=temperature_timeseries,
        )
        fire_timeseries = fm.create_timeseries_fire(
            timeseries_length,
            rng=rng,
            seed=i,
            biomass=100,
            temperature_timeseries=temperature_timeseries,
            humidity_timeseries=humidity_timeseries,
        )

        assert temperature_timeseries.shape == (timeseries_length,)
        assert temperature_timeseries.shape == (timeseries_length,)
        assert fire_timeseries.shape == (timeseries_length,)

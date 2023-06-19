import numpy as np


from fire_forecast import fire_model as fm 




def test():
    """
    Test that the fire time series is created correctly.
    """
    n_tests = 100
    t = 1000
    rng = np.random.default_rng(seed=0)
    for i in range(n_tests):
        fm.create_ts_temperature(t, rng, seed=i)
        fm.create_ts_humidity(t, rng, seed=i)
        fm.create_ts_fire(t, rng=rng, seed=i)
        fm.create_ts_fire(t, rng=rng, seed=i, biomass=100)

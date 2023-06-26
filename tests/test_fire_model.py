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
        ts_temp =fm.create_ts_temperature(t, rng, seed=i)
        ts_hum = fm.create_ts_humidity(t, rng, seed=i)
        fm.create_ts_fire(t, rng=rng, seed=i)
        fm.create_ts_fire(t, rng=rng, seed=i, biomass=100)
        fm.create_ts_fire(t, rng=rng, seed=i, biomass=100, ts_temp=ts_temp)
        fm.create_ts_fire(t, rng=rng, seed=i, biomass=100, ts_hum=ts_hum)
        ts_fire = fm.create_ts_fire(t, rng=rng, seed=i, biomass=100, ts_temp=ts_temp, ts_hum=ts_hum)

        assert ts_temp.shape == (t,)
        assert ts_hum.shape == (t,)
        assert ts_fire.shape == (t,)
"""
Supply a fire model that creates fire time series data. 

The idea is that the forecasting algorithms can be trained on the synthetic.
"""

import numpy as np
import xarray as xr

def create_ts_temperature(t, rng=None, seed=0):
    """
    Create a temperature time series.

    The time series is created for hourly time steps. The temperature includes a diurnal
    cycle and a random trend with additional noise.

    Parameters
    ----------
    t : int
        The length of the time series
    rng : np.random.Generator, optional
        The random number generator used by the function. If not specified, a
        new generator is created.
    seed : int, optional
        The seed for the random number generator. Only used if rng is not
        specified.

    Returns
    -------
    temperature : ndarray
        The time series of the temperature
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    # Create the diurnal cycle
    t_diurnal_cycle = 24
    diurnal_cycle = 5 * np.sin(np.arange(t) * 2 * np.pi / t_diurnal_cycle)
    # Create the random trend
    offset = rng.random() * 10
    trend = np.linspace(offset, offset + rng.random() * 10, t)
    # Create the noise
    noise = rng.normal(0, 2, t)
    # Combine the components
    temperature = diurnal_cycle + trend + noise
    return temperature

def create_ts_humidity(t, rng=None, seed=0):
    """
    Create a humidity time series.

    The time series is created for hourly time steps. The humidity includes a diurnal
    cycle and a random trend with additional noise.

    Parameters
    ----------
    t : int
        The length of the time series
    rng : np.random.Generator, optional
        The random number generator used by the function. If not specified, a
        new generator is created.
    seed : int, optional
        The seed for the random number generator. Only used if rng is not
        specified.

    Returns
    -------
    humidity : ndarray
        The time series of the humidity
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    # Create the diurnal cycle
    t_diurnal_cycle = 24
    diurnal_cycle = 5 * np.sin(np.arange(t) * 2 * np.pi / t_diurnal_cycle)
    # Create the random trend
    offset = rng.random() * 30
    trend = np.linspace(offset, offset + rng.random() * 10, t)
    # Create the noise
    noise = rng.normal(0, 2, t)
    # Combine the components
    humidity = diurnal_cycle + trend + noise + 10
    # Limit to humidity between 0 and 100
    humidity = np.clip(humidity, 1, 100)
    return humidity


def create_ts_fire(t, biomass=None, ts_temp=None, ts_hum=None, rng=None, seed=0):
    """
    Creates a time series of a fire signal.

    Parameters
    ----------
    t : int
        The length of the time series
    biomass : float, optional
        The initial biomass. If not specified, a random value between 0 and 1000
        is used.
    ts_temp : ndarray, optional
        The temperature time series. If not specified, a new time series is
        created.
    ts_hum : ndarray, optional
        The humidity time series. If not specified, a new time series is
        created.
    rng : np.random.Generator, optional
        The random number generator used by the function. If not specified, a
        new generator is created.
    seed : int, optional
        The seed for the random number generator. Only used if rng is not
        specified.

    Returns
    -------
    fire : ndarray
        The time series of the fire signal
    """    
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if ts_temp is None:
        ts_temp = create_ts_temperature(t, rng, seed)
    if ts_hum is None:
        ts_hum = create_ts_humidity(t, rng, seed)

    if biomass is None:
        biomass = rng.random() * 1000
    # Create the fire time series
    ts_fire = np.zeros(t)
    for i in range(1, t):
        df = biomass * ts_temp[i] / 150 * (0.4 - ts_hum[i] / 100) - 0.1
        # Add a random component
        df *= rng.normal(1., 0.5)
        ts_fire[i] = ts_fire[i - 1] + df
        biomass -= ts_fire[i]
        if ts_fire[i] < 0.01:
            ts_fire[i] = 0
            break
    return ts_fire
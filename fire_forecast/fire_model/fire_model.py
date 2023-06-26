"""
Supply a fire model that creates fire time series data. 

The idea is that the forecasting algorithms can be trained on the synthetic data.
"""

# Constants for the fire model
AMPLITUDE_DIURNAL_CYCLE = 5
OFFSET = {"temperature": 10, "humidity": 30}
TREND = {"temperature": 10, "humidity": 10}
NOISE_STD = {"temperature": 2, "humidity": 2, "fire": 0.5}
BASELINE = {"temperature": 0, "humidity": 10}
BIOMASS = 1000
FIRE_SCALE = 1 / 150
FIRE_HUMIDITY_IMPACT = 0.4
FIRE_FORCING = 0.6
FIRE_CUTOFF = 0.01


import numpy as np
import xarray as xr


def create_temperature_timeseries(
    timeseries_length: int, rng: np.random.Generator | None = None, seed: int = 0
) -> np.ndarray:
    """
    Create a temperature time series.

    The time series is created for hourly time steps. The temperature includes a diurnal
    cycle and a random trend with additional noise.

    Args:
        timeseries_length (int): The length of the time series
        rng (np.random.Generator, optional): The random number generator used by the
            function. If not specified, a new generator is created.
        seed (int, optional): The seed for the random number generator. Only used if rng
            is not specified.

    Returns:
        ndarray: The time series of the temperature
    """

    if rng is None:
        rng = np.random.default_rng(seed=seed)
    # Create the diurnal cycle
    t_diurnal_cycle = 24
    diurnal_cycle = AMPLITUDE_DIURNAL_CYCLE * np.sin(
        np.arange(timeseries_length) * 2 * np.pi / t_diurnal_cycle
    )
    # Create the random trend
    offset = rng.random() * OFFSET["temperature"]
    trend = np.linspace(
        offset, offset + rng.random() * TREND["temperature"], timeseries_length
    )
    # Create the noise
    noise = rng.normal(0, NOISE_STD["temperature"], timeseries_length)
    # Combine the components
    temperature = diurnal_cycle + trend + noise + BASELINE["temperature"]
    return temperature


def create_humidity_timeseries(
    timeseries_length: int, rng: np.random.Generator | None = None, seed: int = 0
) -> np.ndarray:
    """
    Create a humidity time series.

    The time series is created for hourly time steps. The humidity includes a diurnal
    cycle and a random trend with additional noise.

    Args:
        timeseries_length (int): The length of the time series
        rng (np.random.Generator, optional): The random number generator used by
            the function. If not specified, a new generator is created.
        seed (int, optional): The seed for the random number generator. Only used if rng is not
            specified.

    Returns:
        ndarray: The time series of the humidity
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    # Create the diurnal cycle
    t_diurnal_cycle = 24
    diurnal_cycle = AMPLITUDE_DIURNAL_CYCLE * np.sin(
        np.arange(timeseries_length) * 2 * np.pi / t_diurnal_cycle
    )
    # Create the random trend
    offset = rng.random() * OFFSET["humidity"]
    trend = np.linspace(
        offset, offset + rng.random() * TREND["humidity"], timeseries_length
    )
    # Create the noise
    noise = rng.normal(0, NOISE_STD["humidity"], timeseries_length)
    # Combine the components
    humidity = diurnal_cycle + trend + noise + BASELINE["humidity"]
    # Limit to humidity between 0 and 100
    humidity = np.clip(humidity, 1, 100)
    return humidity


def create_timeseries_fire(
    timeseries_length: int,
    biomass: float | None = None,
    temperature_timeseries: np.ndarray | None = None,
    humidity_timeseries: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Creates a time series of a fire signal.

    Args:
        timeseries_length  (int): The length of the time series
        biomass (float, optional): The initial biomass. If not specified, a random
            value between 0 and 1000 is used.
        temperature_timeseries (ndarray, optional): The temperature time series. If not
            specified, a new time series is created.
        humidity_timeseries (ndarray, optional): The humidity time series. If not specified, a
            new time series is created.
        rng (np.random.Generator, optional): The random number generator
            used by the function. If not specified, a new generator is created.
        seed (int, optional): The seed for the random number generator.
             Only used if rng is not specified.

    Returns:
        ndarray: The time series of the fire signal
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    if temperature_timeseries is None:
        temperature_timeseries = create_temperature_timeseries(
            timeseries_length, rng, seed
        )
    if humidity_timeseries is None:
        humidity_timeseries = create_humidity_timeseries(timeseries_length, rng, seed)

    if biomass is None:
        biomass = rng.random() * BIOMASS
    # Create the fire time series
    fire_timeseries = np.zeros(timeseries_length)
    for i in range(1, timeseries_length):
        df = (
            biomass
            * temperature_timeseries[i]
            * FIRE_SCALE
            * (FIRE_HUMIDITY_IMPACT - humidity_timeseries[i] / 100)
            - FIRE_FORCING
        )
        # Add a random component
        df *= rng.normal(1.0, NOISE_STD["fire"])
        fire_timeseries[i] = fire_timeseries[i - 1] + df
        biomass -= fire_timeseries[i]
        if fire_timeseries[i] < FIRE_CUTOFF:
            fire_timeseries[i] = 0
            break
    return fire_timeseries

import argparse
import itertools
from pathlib import Path

from loguru import logger

from fire_forecast.deep_learning.iterator import Iterator
from fire_forecast.deep_learning.utils import read_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hyperparameter_config",
        type=str,
        help="Path to the config file.",
    )
    return parser.parse_args()


def collect_parameters(hyper_config: dict) -> list:
    """Collect all combinations of hyperparameters from a config dictionary.

    Args:
        hyper_config (dict): Config dictionary with hyperparameters.

    Returns:
        list: List of all combinations of hyperparameters.
    """
    hyperparameters = []
    keys = []
    for key, value in hyper_config.items():
        if isinstance(value, dict):
            sub_hyperparameters, sub_keys = collect_parameters(value)
            for sub_hyperparameter in sub_hyperparameters:
                hyperparameters.append(sub_hyperparameter)
            for sub_key in sub_keys:
                keys.append([key] + sub_key)
        else:
            hyperparameters.append(value)
            keys.append([key])
    return hyperparameters, keys


def set_value_deep_key(dictionary: dict, keylist: list[str], value):
    if len(keylist) == 1:
        dictionary[keylist[0]] = value
    else:
        dictionary[keylist[0]] = set_value_deep_key(
            dictionary[keylist[0]], keylist[1:], value
        )
    return dictionary


def main():
    args = get_args()
    logger.info(f"Using config file {args.hyperparameter_config}")
    hyper_config = read_config(args.hyperparameter_config)
    base_config = read_config(hyper_config.pop("base_config_path"))
    output_directory = Path(hyper_config.pop("output_directory"))
    hyperparameters, keys = collect_parameters(hyper_config)
    hyperparameter_sets = list(itertools.product(*hyperparameters))
    for i, hyperparameter_set in enumerate(hyperparameter_sets):
        logger.info(f"Hyperparameter set: {hyperparameter_set}")
        config = base_config.copy()
        for key, value in zip(keys, hyperparameter_set):
            # key is a list of keys with arbitrary length. Set the item in the
            # config
            # dictionary by iterating over the keys.
            config = set_value_deep_key(config, key, value)
        logger.info(f"Config: {config}")
        logger.info("Initializing model from config.")
        # TODO: set the output path to a new folder for each hyperparameter set
        output_path = output_directory / f"hyperparameter_set_{i}"
        config["output"]["path"] = str(output_path)
        output_path.mkdir(exist_ok=True)
        iterator = Iterator(config)
        logger.info("Starting training.")
        iterator.train()


if __name__ == "__main__":
    main()

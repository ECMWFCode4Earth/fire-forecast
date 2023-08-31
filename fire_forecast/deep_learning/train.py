import argparse

from loguru import logger

from fire_forecast.deep_learning.iterator import Iterator
from fire_forecast.deep_learning.utils import read_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Path to the config file.",
        default="config.yaml",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(f"Using config file {args.config}")
    config = read_config(args.config)
    logger.info(f"Config: {config}")
    logger.info("Initializing model from config.")
    iterator = Iterator(config)
    logger.info(f"Model initialized.\nModel: {iterator.model}")
    logger.info("Starting training.")
    iterator.train()


if __name__ == "__main__":
    main()

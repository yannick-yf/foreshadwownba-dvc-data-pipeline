"""
This module loads the NBA games training dataset into a MySQL database.

References:
    - https://sebrave.medium.com/how-to-spin-up-a-local-mysql-database-on-macos-a550918f092b
    - https://blog.devart.com/delete-duplicate-rows-in-mysql.html
    - https://numberly.tech/orchestrating-python-workflows-in-apache-airflow-fd8be71ad504
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.utils.logs import get_logger


def load_training_dataset_to_db(config_path: Path) -> None:
    """
    Load the NBA games training dataset into a MySQL database.

    Args:
        config_path (Text): Path to the configuration file.
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)

    # Loading variables from the .env file
    load_dotenv()

    logger = get_logger(
        "LOAD_DATA_TO_DATABASE", log_level=config_params["base"]["log_level"]
    )

    # Read the data to insert into the database
    training_dataset = pd.read_csv("./data/output/nba_games_training_dataset_final_post_cleaned.csv")

    training_dataset.to_csv('./data/final/training_dataset.csv', index=False)

    logger.info("Load Training dataset to the database is complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    load_training_dataset_to_db(config_path=args.config_params)

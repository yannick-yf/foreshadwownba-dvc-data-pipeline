# https://sebrave.medium.com/how-to-spin-up-a-local-mysql-database-on-macos-a550918f092b
# https://blog.devart.com/delete-duplicate-rows-in-mysql.html
# https://numberly.tech/orchestrating-python-workflows-in-apache-airflow-fd8be71ad504

import pymysql
import pandas as pd
from typing import Text
import yaml
import argparse
import os
from pandas.io import sql
from sqlalchemy import create_engine
from src.utils.logs import get_logger
from sqlalchemy import text
import os
from dotenv import load_dotenv
import sys


def get_training_dataset(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    # loading variables from .env file
    load_dotenv()

    logger = get_logger(
        "DOWNLOAD_DATA_FROM_DATABASE", log_level=config_params["base"]["log_level"]
    )

    # Read the data to insert into the db
    nba_games_training_dataset = config_params["get_training_dataset"][
        "nba_games_training_dataset"
    ]
    player_attributes_salaries_dataset = config_params["get_training_dataset"][
        "player_attributes_salaries_dataset"
    ]

    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            user=os.getenv("MYSQL_USERNAME"),
            pw=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            db=os.getenv("MYSQL_DATABASE"),
        )
    )

    nba_games_training_dataset = pd.read_sql(
        """
        SELECT * 
        FROM foreshadwownba.nba_games_training_dataset;""",
        engine,
    )

    nba_games_training_dataset.to_csv(
        "./data/input/nba_games_training_dataset.csv", index=False
    )

    player_attributes_salaries_dataset = pd.read_sql(
        """
        SELECT * 
        FROM foreshadwownba.player_attributes_salaries_dataset;""",
        engine,
    )

    player_attributes_salaries_dataset.to_csv(
        "./data/input/player_attributes_salaries_dataset.csv", index=False
    )

    logger.info("Download data from DataBase complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    get_training_dataset(config_path=args.config_params)

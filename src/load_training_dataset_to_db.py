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


def load_training_dataset_to_db(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    # loading variables from .env file
    load_dotenv()

    logger = get_logger(
        "LOAD_DATA_TO_DATABASE", log_level=config_params["base"]["log_level"]
    )

    # Read the data to insert into the db

    training_dataset = pd.read_csv("./data/output/nba_games_training_dataset_final.csv")

    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            user=os.getenv("MYSQL_USERNAME"),
            pw=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            db=os.getenv("MYSQL_DATABASE"),
        )
    )

    training_dataset.to_sql(
        con=engine, index=False, name="training_dataset", if_exists="append"
    )

    with engine.connect() as conn:
        query1 = text(
            """
            ALTER TABLE training_dataset ADD COLUMN count_ID int(10) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY;
        """
        )
        query2 = text(
            """
            DELETE FROM training_dataset
            WHERE count_ID not in (
                SELECT count_ID FROM (
                    SELECT max(count_ID) as count_ID
                    FROM training_dataset
                    GROUP BY id, tm, opp
                    ) as c
                );
        """
        )
        query3 = text(
            """
            ALTER TABLE training_dataset DROP count_ID;
        """
        )
        conn.execute(query1)
        conn.execute(query2)
        conn.execute(query3)

    logger.info("Load Training dataset to Database complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    load_training_dataset_to_db(config_path=args.config_params)

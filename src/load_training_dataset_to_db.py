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

    engine = create_engine(
        f"mysql+pymysql://{os.getenv('MYSQL_USERNAME')}:{os.getenv('MYSQL_PASSWORD')}"
        f"@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DATABASE')}"
    )

    training_dataset.to_sql(
        con=engine, index=False, name="training_dataset", if_exists="append"
    )

    with engine.connect() as conn:
        query1 = text(
            """
            ALTER TABLE training_dataset 
            ADD COLUMN count_ID int(10) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY;
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

    logger.info("Load Training dataset to the database is complete.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    load_training_dataset_to_db(config_path=args.config_params)

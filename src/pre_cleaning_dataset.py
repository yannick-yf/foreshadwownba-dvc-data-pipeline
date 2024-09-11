import pandas as pd
import numpy as np
from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger


def pre_cleaning_dataset(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "PRE_CLEANED_DATASET", log_level=config_params["base"]["log_level"]
    )

    nba_games_training_dataset = pd.read_csv(
        "./data/input/nba_games_training_dataset.csv"
    )
    logger.info("Shape of the Dataframe " + str(nba_games_training_dataset.shape))

    nba_games_training_dataset["overtime"] = nba_games_training_dataset[
        "overtime"
    ].fillna("NOT")

    # Delete games during Covid in the Bubble
    nba_games_training_dataset = nba_games_training_dataset.drop(
        nba_games_training_dataset[
            (nba_games_training_dataset["game_date"] > "2020-03-10")
            & (nba_games_training_dataset["id_season"] == 2020)
        ].index
    )

    # Column Selection
    column_to_select = [
        "id_season",
        "id",
        "game_nb",
        "game_date",
        "extdom",
        "tm",
        "opp",
        "results",
        "pts_tm",
        "pts_opp",
        "w_tot",
        "overtime",
        "streak_w_l",
    ]

    nba_games_training_dataset = nba_games_training_dataset[column_to_select]

    logger.info("Shape of the Dataframe " + str(nba_games_training_dataset.shape))
    nba_games_training_dataset.to_csv(
        "./data/processed/nba_games_training_dataset_pre_cleaned.csv", index=False
    )

    logger.info("Pre Cleaned NBA games data step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    pre_cleaning_dataset(config_path=args.config_params)

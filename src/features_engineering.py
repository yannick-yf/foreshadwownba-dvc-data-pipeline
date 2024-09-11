import pandas as pd
import numpy as np

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger

from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features,
    previous_games_ratio_average_features,
    previous_season_ratio_features,
)
from src.feature_engineering_functions.rest_days_between_games import (
    calculate_rest_days_between_games,
)
from src.feature_engineering_functions.duration_trip_features import (
    duration_trip_hours_between_cities,
)
from src.feature_engineering_functions.previous_days_average_features import (
    previous_days_average_features,
)
from src.feature_engineering_functions.games_date_processing import (
    game_on_weekend_features,
    extract_days_of_week_from_date,
)
from src.feature_engineering_functions.average_ratio_win_loose_ext_game import (
    average_ratio_win_loose_ext_game,
)
from src.feature_engineering_functions.last_game_overtime import last_game_overtime
from src.feature_engineering_functions.streack_w_l import calculate_streak_features
from src.feature_engineering_functions.final_cleaning import final_cleaning


def copy_df(df):
    return df.copy()


def features_engineering_pipeline(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "LAST_GAMES_RATIO_AVERAGE_FEATURES",
        log_level=config_params["base"]["log_level"],
    )

    # Read the input data for the step
    training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
    )

    training_dataset_w_features = (
        training_dataset.pipe(copy_df)
        .pipe(previous_games_average_features)
        .pipe(previous_games_ratio_average_features)
        .pipe(previous_season_ratio_features)
        .pipe(calculate_rest_days_between_games)
        .pipe(duration_trip_hours_between_cities)
        .pipe(previous_days_average_features)
        .pipe(extract_days_of_week_from_date)
        .pipe(game_on_weekend_features)
        .pipe(average_ratio_win_loose_ext_game)
        .pipe(last_game_overtime)
        .pipe(calculate_streak_features)
        .pipe(final_cleaning)
    )

    # Save the data
    logger.info(
        "Final Shape of the Dataframe after featuer engineering process "
        + str(training_dataset_w_features.shape)
    )
    training_dataset_w_features.to_csv(
        "./data/processed/nba_games_training_dataset_cleaned_w_features.csv",
        index=False,
    )

    logger.info("Feateure Engineering Generation step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    features_engineering_pipeline(config_path=args.config_params)

"""
This module performs feature engineering on the NBA games training dataset.
"""

import argparse
from pathlib import Path
from warnings import simplefilter

import pandas as pd
import yaml

from src.feature_engineering_functions.average_ratio_win_loose_ext_game import (
    average_ratio_win_loose_ext_game,
)
from src.feature_engineering_functions.duration_trip_features import (
    duration_trip_hours_between_cities,
)
from src.feature_engineering_functions.final_cleaning import final_cleaning
from src.feature_engineering_functions.games_date_processing import (
    extract_days_of_week_from_date,
    game_on_weekend_features,
)
from src.feature_engineering_functions.games_streak_features import (
    calculate_streak_features,
)
from src.feature_engineering_functions.last_game_overtime import last_game_overtime
from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features,
    previous_games_ratio_average_features,
    previous_season_ratio_features,
)
from src.feature_engineering_functions.previous_days_average_features import (
    previous_days_average_features,
)
from src.feature_engineering_functions.rest_days_between_games import (
    calculate_rest_days_between_games,
)
from src.feature_engineering_functions.handle_categorical_features import (
    handle_categorical_features,
)
from src.utils.logs import get_logger

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def copy_df(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a copy of the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A copy of the input DataFrame.
    """
    return training_df.copy()


def features_engineering_pipeline(config_path: Path) -> pd.DataFrame:
    """
    Perform feature engineering on the NBA games training dataset.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        pd.DataFrame: The training dataset with engineered features.
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "FEATURES_ENGINEERING_PROESS",
        log_level=config_params["base"]["log_level"],
    )

    # Read the input data for the step
    training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
    )

    training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

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
        .pipe(handle_categorical_features)
        .pipe(final_cleaning)
    )

    # Save the data
    logger.info(
        "Final Shape of the DataFrame after feature engineering process %s",
        str(training_dataset_w_features.shape),
    )
    training_dataset_w_features.to_csv(
        "./data/processed/nba_games_training_dataset_cleaned_w_features.csv",
        index=False,
    )

    logger.info("Feature Engineering Generation step complete")
    return training_dataset_w_features


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()

    features_engineering_pipeline(config_path=args.config_params)

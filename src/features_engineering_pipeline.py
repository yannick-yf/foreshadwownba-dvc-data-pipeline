"""
This module performs feature engineering on the NBA games training dataset.
"""

import argparse
from pathlib import Path
from warnings import simplefilter

import pandas as pd
import yaml
import os

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
from src.feature_engineering_functions.opponent_features import (
    get_opponent_features,
)
from src.utils.logs import get_logger

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = get_logger(
    "FEATURES_ENGINEERING_PROGRESS",
    log_level='INFO',
)

def copy_df(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a copy of the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A copy of the input DataFrame.
    """
    return training_df.copy()

def features_engineering_pipeline(
        input_file_folder_name: str = 'data/processed/nba_games_training_dataset_final.csv',
        output_file_folder_name: str ='data/output/nba_games_training_dataset_final.csv'
        ) -> None:
    """
    Feature Engineering Pipeline.

    Args:
        input_file_folder_name (str): Pre-cleaned DataFrame 
        output_file_folder_name (str): DataFrame with features computed.
    """

    # Read the input data for the step
    training_dataset = pd.read_csv(
        input_file_folder_name
    )

    training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

    # Column to process for previous_games_average_features

    training_dataset_w_features = (
        training_dataset.pipe(copy_df)
        .pipe(previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"])
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
        .pipe(get_opponent_features)
        .pipe(final_cleaning)
    )

    # Save the data
    logger.info(
        "Final Shape of the DataFrame after feature engineering process %s",
        str(training_dataset_w_features.shape),
    )
    training_dataset_w_features.to_csv(
        output_file_folder_name,
        index=False,
    )

    logger.info("Feature Engineering Generation step complete")

def get_args():
    """
    Parse command line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    _dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params-file",
        type=Path,
        default="params.yaml",
    )

    args, _ = parser.parse_known_args()
    params = yaml.safe_load(args.params_file.open())

    pre_cleaning_dataset_params = params["pre_cleaning_dataset"]
    features_engineering_pipeline_params = params["features_engineering_pipeline"]
    
    input_file_folder_name = os.path.join(
        pre_cleaning_dataset_params['output_folder'], 
        pre_cleaning_dataset_params['output_file_name'] +  ".csv"
        )
    
    parser.add_argument(
        "--input-file-folder-name",
        dest="input_file_folder_name",
        type=str,
        default=input_file_folder_name,
    )

    output_file_folder_name = os.path.join(
        features_engineering_pipeline_params['output_file'] + '.csv')
    
    parser.add_argument(
        "--output-file-folder-name",
        dest="output_file_folder_name",
        type=Path,
        default=output_file_folder_name,
    )

    args = parser.parse_args()
        
    return args

def main():
    """Run the Feature Engineering Step."""
    args = get_args()

    features_engineering_pipeline(
        input_file_folder_name=args.input_file_folder_name,
        output_file_folder_name=args.output_file_folder_name
    )

if __name__ == "__main__":
    main()
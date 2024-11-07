"""
This module performs feature engineering on the NBA games training dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logs import get_logger

logger = get_logger(
    "PRE_CLEANED_DATASET", log_level='INFO'
)

def best_team_name(row) -> str:
    """
    Determine the best team name based on various features.

    Args:
        row (pd.Series): A row from the training dataset.

    Returns:
        str: The name of the best team.
    """
    if row["before_average_W_ratio"] > row["before_average_W_ratio"]:
        val = row["tm"]
    elif row["before_average_W_ratio"] < row["before_average_W_ratio"]:
        val = row["opp"]
    else:
        if (
            row["before_average_lasttengame_W_ratio"]
            > row["before_average_lasttengame_W_ratio"]
        ):
            val = row["tm"]
        elif (
            row["before_average_lasttengame_W_ratio"]
            < row["before_average_lasttengame_W_ratio"]
        ):
            val = row["opp"]
        else:
            if (
                row["before_average_lastfivegame_W_ratio"]
                > row["before_average_lastfivegame_W_ratio"]
            ):
                val = row["tm"]
            elif (
                row["before_average_lastfivegame_W_ratio"]
                < row["before_average_lastfivegame_W_ratio"]
            ):
                val = row["opp"]
            else:
                if row["before_streak_w_l"] > row["before_streak_w_l"]:
                    val = row["tm"]
                elif row["before_streak_w_l"] < row["before_streak_w_l"]:
                    val = row["opp"]
                else:
                    if row["before_season_ratio"] > row["before_season_ratio"]:
                        val = row["tm"]
                    elif row["before_season_ratio"] < row["before_season_ratio"]:
                        val = row["opp"]
                    else:
                        if row["rest"] > row["rest"]:
                            val = row["tm"]
                        elif row["rest"] < row["rest"]:
                            val = row["opp"]
                        else:
                            if row["extdom"] == "dom":
                                val = row["tm"]
                            else:
                                val = row["opp"]
    return val

def get_y_variables(
        input_file_folder_name: str = 'data/processed/nba_games_training_dataset_cleaned_w_features.csv',
        output_file_folder_name: str = 'data/processed/nba_games_training_dataset_final.csv',
        ) -> pd.DataFrame:
    """
    Get y variables for the ML models

    Args:
        input_file (str): Name and path to the input file
        output_file (str): Name and path to the output file
    """

    training_df = pd.read_csv(
        input_file_folder_name
    )

    training_df["results"] = np.where(
        training_df["pts_tm"] > training_df["pts_opp"], "W", "L"
    )
    training_df["results"] = training_df["results"].astype(str)
    training_df["name_best_team"] = training_df.apply(best_team_name, axis=1)

    training_df["results"] = np.where(
        training_df["pts_tm"] > training_df["pts_opp"], 1, 0
    )
    training_df["y_prob_win"] = np.where(training_df["results"] == 1, "W", "L")
    training_df["y_prob_win"] = np.where(training_df["y_prob_win"] == "W", "1", "0")

    training_df["name_win_team"] = np.where(
        training_df["pts_tm"] > training_df["pts_opp"],
        training_df["tm"],
        training_df["opp"],
    )

    training_df["y_bestworst"] = np.where(
        training_df["name_best_team"] == training_df["name_win_team"], 1, 0
    )

    training_df.to_csv(
        output_file_folder_name, index=False
    )

    logger.info("Get Y variables step complete")


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

    get_y_variables_params = params["get_y_variables"]

    input_file_folder_name = get_y_variables_params['input_file'] +\
        '.csv'
    
    parser.add_argument(
        "--input-file-folder-name",
        dest="input_file_folder_name",
        type=str,
        default=input_file_folder_name,
    )

    output_file_folder_name = get_y_variables_params['output_file'] +\
        '.csv'
    
    parser.add_argument(
        "--output-file-folder-name",
        dest="output_file_folder_name",
        type=str,
        default=output_file_folder_name,
    )

    args = parser.parse_args()

    return args

def main():
    """Run the Pre Train Multiple Models Pipeline."""
    args = get_args()

    get_y_variables(
        input_file_folder_name = args.input_file_folder_name,
        output_file_folder_name = args.output_file_folder_name
    )

if __name__ == "__main__":
    main()
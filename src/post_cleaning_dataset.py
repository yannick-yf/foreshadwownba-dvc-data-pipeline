"""
This module performs pre-cleaning on the NBA games training dataset.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import yaml


from src.utils.logs import get_logger

logger = get_logger("POST_CLEANED_DATASET", log_level="INFO")


def post_cleaning_dataset(
    input_file_folder_name: str = "data/processed/nba_games_training_dataset_final.csv",
    output_training_file_folder_name: str = "data/output/nba_games_training_dataset_final.csv",
    output_inseason_file_folder_name: str = "data/output/nba_games_inseason_dataset_final.csv",
) -> None:
    """
    Post Cleaning Dataset.

    Args:
        input_file_folder_name (str): Unified gamelogs and schedules for multiple seasons
        output_training_file_folder_name (str): Path where to save the training dataframe.
        output_inseason_file_folder_name (str): Path where to save the inseason dataframe.
    """

    nba_games_training_dataset = pd.read_csv(input_file_folder_name)

    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    # Column Selection
    nba_games_training_dataset = nba_games_training_dataset.drop(
        [
            "w_tot",
            "overtime",
            "streak_w_l",
            "y_bestworst",
            "name_best_team",
            "y_prob_win",
            "name_win_team",
            "day_of_week",
            "extdom",
            "week_weekend",
            "last_game_overtime",
            "tm_opp",
            "opp_opp",
            "pts_tm",
            "pts_opp",
        ],
        axis=1,
        errors="ignore",
    )

    # game_nb as integer
    nba_games_training_dataset["game_nb"] = nba_games_training_dataset[
        "game_nb"
    ].astype("Int64")

    # Remove game one for each team because features cannot be computed for game 1
    nba_games_training_dataset = nba_games_training_dataset[
        nba_games_training_dataset["game_nb"] != 1
    ]

    # Remove playoffs games and keep missing game value for the inseason
    nba_games_training_dataset_inseason = nba_games_training_dataset[
        nba_games_training_dataset["id_season"] == 2025
    ]

    logger.info(
        "Shape of the Training DataFrame %s", str(nba_games_training_dataset.shape)
    )
    logger.info(
        "Shape of the Inseason DataFrame %s",
        str(nba_games_training_dataset_inseason.shape),
    )

    nba_games_training_dataset.to_csv(output_training_file_folder_name, index=False)
    nba_games_training_dataset_inseason.to_csv(
        output_inseason_file_folder_name, index=False
    )

    logger.info("Post Cleaned NBA games data step complete")


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
    post_cleaning_dataset_params = params["post_cleaning_dataset"]

    output_training_file_folder_name = os.path.join(
        post_cleaning_dataset_params["output_folder"],
        post_cleaning_dataset_params["output_training_file_name"] + ".csv",
    )

    output_inseason_file_folder_name = os.path.join(
        post_cleaning_dataset_params["output_folder"],
        post_cleaning_dataset_params["output_inseason_file_name"] + ".csv",
    )

    input_file_folder_name = os.path.join(
        get_y_variables_params["output_file"] + ".csv"
    )

    parser.add_argument(
        "--input-file-folder-name",
        dest="input_file_folder_name",
        type=str,
        default=input_file_folder_name,
    )

    parser.add_argument(
        "--output-training-file-folder-name",
        dest="output_training_file_folder_name",
        type=Path,
        default=output_training_file_folder_name,
    )

    parser.add_argument(
        "--output-inseason-file-folder-name",
        dest="output_inseason_file_folder_name",
        type=Path,
        default=output_inseason_file_folder_name,
    )

    parser.add_argument(
        "--output-folder",
        dest="output_folder",
        type=Path,
        default=post_cleaning_dataset_params["output_folder"],
    )

    args = parser.parse_args()

    args.output_folder.parent.mkdir(parents=True, exist_ok=True)

    return args


def main():
    """Run the Post Cleaning Dataset Step."""
    args = get_args()

    post_cleaning_dataset(
        input_file_folder_name=args.input_file_folder_name,
        output_training_file_folder_name=args.output_training_file_folder_name,
        output_inseason_file_folder_name=args.output_inseason_file_folder_name,
    )


if __name__ == "__main__":
    main()

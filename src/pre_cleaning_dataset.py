"""
This module performs pre-cleaning on the NBA games training dataset.
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml
import os

from src.utils.logs import get_logger

logger = get_logger(
    "PRE_CLEANED_DATASET", log_level='INFO'
)

def pre_cleaning_dataset(
        input_file_folder_name: str ='data/input/nba_gamelog_schedule_dataset.csv',
        output_folder: Path ='data/processed',
        output_file_name: str = 'nba_games_training_dataset_pre_cleaned'
        ) -> None:
    """
    Pre Cleaning Dataset.

    Args:
        input_file_folder_name (str): Unified gamelogs and schedules for multiple seasons
        output_folder (Path): Path where to save the cleaned unified dataframe.
        output_file_name (str): Name of the cleaned unfied dataframe
    """

    nba_games_training_dataset = pd.read_csv(
        input_file_folder_name
    )
    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    #-------------------------------------------
    # Check for Duplciate and raise info/warnings
    nb_duplicated_rows = nba_games_training_dataset.duplicated(
        subset=['id_season','tm','game_date'],
        keep='first').sum()

    if nb_duplicated_rows > 0:
        logger.info('DUPLICATED ROWS IN THE DATAFRAME')

    nba_games_training_dataset = nba_games_training_dataset.drop_duplicates(
        subset=['id_season', 'tm', 'game_date']
        )

    # Overtime features
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
    columns_to_select = [
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

    nba_games_training_dataset = nba_games_training_dataset[columns_to_select]

    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    nba_games_training_dataset.to_csv(
        os.path.join(output_folder, output_file_name +  ".csv"), index=False
    )

    logger.info("Pre Cleaned NBA games data step complete")

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
    pre_gamelog_schedule_unification_params = params["gamelog_schedule_unification"]

    input_file_folder_name = os.path.join(
        pre_gamelog_schedule_unification_params['unified_file_path'], 
        pre_gamelog_schedule_unification_params['unified_file_name'] +  ".csv"
        )
    
    parser.add_argument(
        "--input-file-folder-name",
        dest="input_file_folder_name",
        type=Path,
        default=input_file_folder_name,
    )

    parser.add_argument(
        "--output-folder",
        dest="output_folder",
        type=Path,
        default=pre_cleaning_dataset_params["output_folder"],
    )

    parser.add_argument(
        "--output-file-name",
        dest="output_file_name",
        type=str,
        default=pre_cleaning_dataset_params["output_file_name"],
    )

    args = parser.parse_args()

    args.output_folder.parent.mkdir(
        parents=True, 
        exist_ok=True)
        
    return args

def main():
    """Run the Pre Train Multiple Models Pipeline."""
    args = get_args()

    pre_cleaning_dataset(
        input_file_folder_name=args.input_file_folder_name,
        output_folder=args.output_folder,
        output_file_name=args.output_file_name
    )

if __name__ == "__main__":
    main()
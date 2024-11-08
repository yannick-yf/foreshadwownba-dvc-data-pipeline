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
    "POST_CLEANED_DATASET", log_level='INFO'
)

def post_cleaning_dataset(
        input_file_folder_name: str = 'data/processed/nba_games_training_dataset_final.csv',
        output_file_folder_name: str ='data/output/nba_games_training_dataset_final.csv'
        ) -> None:
    """
    Post Cleaning Dataset.

    Args:
        input_file_folder_name (str): Unified gamelogs and schedules for multiple seasons
        output_file_folder_name (str): Path where to save the cleaned unified dataframe.
    """

    nba_games_training_dataset = pd.read_csv(
        input_file_folder_name
        )

    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    # Column Selection
    nba_games_training_dataset = nba_games_training_dataset.drop([
            'game_date', 
            'w_tot', 
            'overtime', 
            'streak_w_l', 
            'y_bestworst', 
            'name_best_team',	
            'y_prob_win',	
            'name_win_team',
            'day_of_week'], 
        axis=1,
        errors='ignore')
    
    nba_games_training_dataset = nba_games_training_dataset.dropna()
    
    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))


    nba_games_training_dataset.to_csv(
        output_file_folder_name, index=False
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

    output_file_folder_name = os.path.join(
        post_cleaning_dataset_params['output_folder'], 
        post_cleaning_dataset_params['output_file_name'] +  ".csv"
        )
    
    parser.add_argument(
        "--input-file-folder-name",
        dest="input_file_folder_name",
        type=str,
        default=get_y_variables_params["output_file"],
    )
    
    parser.add_argument(
        "--output-file-folder-name",
        dest="output_file_folder_name",
        type=Path,
        default=output_file_folder_name,
    )

    parser.add_argument(
        "--output-folder",
        dest="output_folder",
        type=Path,
        default=post_cleaning_dataset_params["output_folder"],
    )

    args = parser.parse_args()

    args.output_folder.parent.mkdir(
        parents=True, 
        exist_ok=True)
        
    return args

def main():
    """Run the Post Cleaning Dataset Step."""
    args = get_args()

    post_cleaning_dataset(
        input_file_folder_name=args.input_file_folder_name,
        output_file_folder_name=args.output_file_folder_name
    )

if __name__ == "__main__":
    main()
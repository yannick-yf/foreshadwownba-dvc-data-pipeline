"""
This module performs pre-cleaning on the NBA games training dataset.
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml

from src.utils.logs import get_logger


def post_cleaning_dataset(config_path: Path) -> pd.DataFrame:
    """
    Load and pre-clean the NBA games training dataset.

    Args:
        config_path (Text): Path to the configuration file.

    Returns:
        pd.DataFrame: The pre-cleaned training dataset.
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "POST_CLEANED_DATASET", log_level=config_params["base"]["log_level"]
    )

    nba_games_training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_final.csv"
    )
    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    # Column Selection
    nba_games_training_dataset = nba_games_training_dataset.drop([
            'game_date', 
            'tm', 
            'opp', 
            'w_tot', 
            'overtime', 
            'streak_w_l', 
            'y_bestworst', 
            'name_best_team',	
            'y_prob_win',	
            'name_win_team',
            'day_of_week'], 
        axis=1)
    
    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    nba_games_training_dataset.to_csv(
        "./data/output/nba_games_training_dataset_final_post_cleaned.csv", index=False
    )

    logger.info("Post Cleaned NBA games data step complete")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()
    post_cleaning_dataset(config_path=args.config_params)

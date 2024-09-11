import pandas as pd
import numpy as np
from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger


def best_team_name(row):
    if (row['before_average_W_ratio'] > row['before_average_W_ratio']) :
        val = row['tm']
    elif (row['before_average_W_ratio'] < row['before_average_W_ratio']):
        val = row['opp']
    else:
        if (row['before_average_lasttengame_W_ratio'] > row['before_average_lasttengame_W_ratio']):
            val = row['tm']
        elif (row['before_average_lasttengame_W_ratio'] < row['before_average_lasttengame_W_ratio']):
            val = row['opp']
        else:
            if (row['before_average_lastfivegame_W_ratio'] > row['before_average_lastfivegame_W_ratio']):
                val = row['tm']
            elif (row['before_average_lastfivegame_W_ratio'] < row['before_average_lastfivegame_W_ratio']):
                val = row['opp']
            else:
                if (row['before_streak_w_l'] > row['before_streak_w_l']):
                    val = row['tm']
                elif (row['before_streak_w_l'] < row['before_streak_w_l']):
                    val = row['opp']
                else:
                    if (row['before_season_ratio'] > row['before_season_ratio']):
                        val = row['tm']
                    elif (row['before_season_ratio'] < row['before_season_ratio']):
                        val = row['opp']
                    else:
                        if (row['rest'] > row['rest']):
                            val = row['tm']
                        elif (row['rest'] < row['rest']):
                            val = row['opp']
                        else:
                            if row['extdom'] == 'dom':
                                val = row['tm']
                            else:
                                val = row['opp']
    return val

def get_variables(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "PRE_CLEANED_DATASET", log_level=config_params["base"]["log_level"]
    )

    training_df = pd.read_csv(
        "./data/processed/nba_games_training_dataset_cleaned_w_features.csv"
    )

    training_df['results'] = np.where(
        training_df['pts_tm'] > training_df['pts_opp'],
        'W',
        'L')

    training_df['results'] = training_df['results'].astype(str)
    training_df['name_best_team'] = training_df.apply(best_team_name, axis=1)

    training_df['results'] = np.where(
        training_df['pts_tm'] > training_df['pts_opp'],
        1,
        0)

    training_df['y_prob_win'] = np.where(
        training_df['results'] == 1,
        'W',
        'L')

    training_df['y_prob_win'] = np.where(
        training_df['y_prob_win'] == 'W',
        '1',
        '0')

    training_df['name_win_team'] = np.where(
        training_df['pts_tm'] > training_df['pts_opp'],
        training_df['tm'], training_df['opp'])

    training_df['y_bestworst'] = np.where(
        training_df['name_best_team'] == training_df['name_win_team'],
        1,
        0)

    training_df.to_csv(
        "./data/processed/nba_games_training_dataset_final.csv", index=False
    )

    logger.info("Pre Cleaned NBA games data step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    get_variables(config_path=args.config_params)

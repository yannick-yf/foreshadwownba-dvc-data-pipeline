import pandas as pd
import numpy as np
from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger


def last_games_ratio_average_features(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "LAST_GAMES_RATIO_AVERAGE_FEATURES", log_level=config_params["base"]["log_level"]
    )

    # Read the input data for the step
    training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_elo_features.csv"
    )

    new_training_dataset = training_dataset.copy()

    df = new_training_dataset[['id_season', 'game_nb', 'game_date', 'extdom', 'tm', 'opp', 'results']]
    df['w_for_ratio'] = np.where(df['results']=='W',1,0)
    df['cum_sum_win'] = df.groupby(['id_season','tm'])['w_for_ratio'].cumsum()
    df['average_ratio'] = df['cum_sum_win'] / df['game_nb']

    df['before_average_W_ratio'] = df.groupby(['id_season','tm'])['average_ratio'].shift(1)

    df_ratio_last_5_games = df.set_index('game_date').groupby(['id_season','tm']).rolling(5)['w_for_ratio'].sum().reset_index()
    df_ratio_last_5_games['before_average_lastfivegame_W_ratio'] = df_ratio_last_5_games['w_for_ratio'] / 5
    df_ratio_last_5_games['before_average_lastfivegame_W_ratio'] = df_ratio_last_5_games.groupby(['id_season','tm'])['before_average_lastfivegame_W_ratio'].shift(1)

    df_ratio_last_10_games = df.set_index('game_date').groupby(['id_season','tm']).rolling(10)['w_for_ratio'].sum().reset_index()
    df_ratio_last_10_games['before_average_lasttengame_W_ratio'] = df_ratio_last_10_games['w_for_ratio'] / 10
    df_ratio_last_10_games['before_average_lasttengame_W_ratio'] = df_ratio_last_10_games.groupby(['id_season','tm'])['before_average_lasttengame_W_ratio'].shift(1)

    df_ratio_last =  pd.merge(
        df_ratio_last_5_games,
        df_ratio_last_10_games,
        how='left',
        left_on=['id_season', 'game_date', 'tm'],
        right_on=['id_season', 'game_date', 'tm'])

    final =  pd.merge(
        df,
        df_ratio_last,
        how='left',
        left_on=['id_season', 'game_date', 'tm'],
        right_on=['id_season', 'game_date', 'tm'])

    # Fill na values
    final['before_average_lastfivegame_W_ratio'] = final['before_average_lastfivegame_W_ratio' ].fillna(final['before_average_W_ratio'])
    final['before_average_lasttengame_W_ratio'] = final['before_average_lasttengame_W_ratio'].fillna(final['before_average_lastfivegame_W_ratio'])
    final['before_average_lasttengame_W_ratio'] = final['before_average_lasttengame_W_ratio'].fillna(final['before_average_W_ratio'])

    final['diff_all_minus_lastfivegame_W_ratio'] = final['before_average_W_ratio'] - final['before_average_lastfivegame_W_ratio']
    final['diff_all_minus_lasttengame_W_ratio' ] = final['before_average_W_ratio'] - final['before_average_lasttengame_W_ratio']

    final = final.drop(['game_nb','extdom', 'results', 'w_for_ratio', 'cum_sum_win', 'average_ratio'	,'w_for_ratio_x', 'w_for_ratio_y'], axis = 1)

    new_training_dataset =  pd.merge(
        new_training_dataset,
        final,
        how='left',
        left_on=['id_season', 'game_date', 'tm', 'opp'],
        right_on=['id_season', 'game_date', 'tm', 'opp'])

    # Save the data
    new_training_dataset.to_csv(
        "./data/processed/nba_games_training_dataset_ratio_average_features.csv",
        index=False,
    )

    logger.info("Last Games Ratio Average Features step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    last_games_ratio_average_features(config_path=args.config_params)

from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features, previous_games_win_ratio_average_features,
)

class TestPreviousGamesRatioFeatures(TestCase):
    def setUp(self) -> None:
        self.season = 2018
        self.team = "CLE"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_previous_games_ratio_average_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN previous_games_ratio_average_features is applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_last_game_W_ratio' features is correctly computed
            - Verify that 'before_average_last_game_W_ratio' for the first game is NaN, indicating no prior games.
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        training_df_w_previous_games_ratio_features = (training_dataset
            .pipe(previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"])
            .pipe(previous_games_win_ratio_average_features)
        )

        training_df_w_previous_games_ratio_features = (
            training_df_w_previous_games_ratio_features[
                (training_df_w_previous_games_ratio_features["tm"] == self.team)
                & (
                    training_df_w_previous_games_ratio_features["id_season"]
                    == self.season
                )
            ]
        )

        before_average_lastfivegame_W_ratio_first_game = training_df_w_previous_games_ratio_features["before_average_lastfivegame_W_ratio"][
            training_df_w_previous_games_ratio_features["game_nb"] == 1
        ].values[0]

        before_average_lasttengame_W_ratio_first_game = training_df_w_previous_games_ratio_features["before_average_lasttengame_W_ratio"][
            training_df_w_previous_games_ratio_features["game_nb"] == 1
        ].values[0]

        before_average_lastfivegame_W_ratio_fifth_game = training_df_w_previous_games_ratio_features["before_average_lastfivegame_W_ratio"][
            training_df_w_previous_games_ratio_features["game_nb"] == 6
        ].values[0]

        true_before_average_lastfivegame_W_ratio_value = np.count_nonzero(training_df_w_previous_games_ratio_features.iloc[0:5]['results'].values == 'W') / 5

        assert before_average_lastfivegame_W_ratio_fifth_game == true_before_average_lastfivegame_W_ratio_value
        self.assertTrue(np.isnan(before_average_lastfivegame_W_ratio_first_game))
        self.assertTrue(np.isnan(before_average_lasttengame_W_ratio_first_game))
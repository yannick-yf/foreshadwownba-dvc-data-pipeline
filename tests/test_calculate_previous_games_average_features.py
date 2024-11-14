from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features,
)


class TestLastGameOvertime(TestCase):
    def setUp(self) -> None:
        self.season = 2024
        self.team = "BOS"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_before_average_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN previous_games_average_features is applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_pts_tm' for the second game matches the points of the first game.
            - Verify that 'before_average_pts_tm' for the first game is NaN, indicating no prior games.
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        training_df_w_previous_games_average_features_w = training_dataset.pipe(
            previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"]
        )

        training_df_w_previous_games_average_features_w = (
            training_df_w_previous_games_average_features_w[
                (training_df_w_previous_games_average_features_w["tm"] == self.team)
                & (
                    training_df_w_previous_games_average_features_w["id_season"]
                    == self.season
                )
            ]
        )

        pts_tm_first_game = training_df_w_previous_games_average_features_w["pts_tm"][
            training_df_w_previous_games_average_features_w["game_nb"] == 1
        ].values[0]

        before_average_pts_tm = training_df_w_previous_games_average_features_w[
            "before_average_pts_tm"
        ][training_df_w_previous_games_average_features_w["game_nb"] == 2].values[0]

        before_average_pts_tm_first_game = (
            training_df_w_previous_games_average_features_w["before_average_pts_tm"][
                training_df_w_previous_games_average_features_w["game_nb"] == 1
            ].values[0]
        )

        assert before_average_pts_tm == pts_tm_first_game
        self.assertTrue(np.isnan(before_average_pts_tm_first_game))

    def test_before_average_lastfivegame_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_streak_features and previous_games_average_features are applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_pts_tm' for the second game matches the points of the first game.
            - Verify that 'before_average_pts_tm' for the first game is NaN, indicating no prior games.
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        training_df_w_previous_games_average_features_w = training_dataset.pipe(
            previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"]
        )

        training_df_w_previous_games_average_features_w = (
            training_df_w_previous_games_average_features_w[
                (training_df_w_previous_games_average_features_w["tm"] == self.team)
                & (
                    training_df_w_previous_games_average_features_w["id_season"]
                    == self.season
                )
            ]
        )

        pts_tm_first_5_games_avg = training_df_w_previous_games_average_features_w[
            "pts_tm"
        ][training_df_w_previous_games_average_features_w["game_nb"] <= 5].mean()

        before_average_lastfivegame_pts_tm = (
            training_df_w_previous_games_average_features_w[
                "before_average_lastfivegame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 6].values[0]
        )

        assert before_average_lastfivegame_pts_tm == pts_tm_first_5_games_avg

    def test_before_average_lasttengame_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_streak_features and previous_games_average_features are applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_pts_tm' for the second game matches the points of the first game.
            - Verify that 'before_average_pts_tm' for the first game is NaN, indicating no prior games.
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        training_df_w_previous_games_average_features_w = training_dataset.pipe(
            previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"]
        )

        training_df_w_previous_games_average_features_w = (
            training_df_w_previous_games_average_features_w[
                (training_df_w_previous_games_average_features_w["tm"] == self.team)
                & (
                    training_df_w_previous_games_average_features_w["id_season"]
                    == self.season
                )
            ]
        )

        pts_tm_10_games_avg = training_df_w_previous_games_average_features_w["pts_tm"][
            (training_df_w_previous_games_average_features_w["game_nb"] <= 20)
            & (training_df_w_previous_games_average_features_w["game_nb"] > 10)
        ].mean()

        before_average_lasttengame_pts_tm = (
            training_df_w_previous_games_average_features_w[
                "before_average_lasttengame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 21].values[
                0
            ]
        )

        assert before_average_lasttengame_pts_tm == pts_tm_10_games_avg

    def test_fillna_before_average_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_streak_features and previous_games_average_features are applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_pts_tm' for the second game matches the points of the first game.
            - Verify that 'before_average_pts_tm' for the first game is NaN, indicating no prior games.
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        # Column to process for previous_games_average_features

        training_df_w_previous_games_average_features_w = training_dataset.pipe(
            previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"]
        )

        training_df_w_previous_games_average_features_w = (
            training_df_w_previous_games_average_features_w[
                (training_df_w_previous_games_average_features_w["tm"] == self.team)
                & (
                    training_df_w_previous_games_average_features_w["id_season"]
                    == self.season
                )
            ]
        )

        before_average_game_pts_tm_1 = training_df_w_previous_games_average_features_w[
            "before_average_pts_tm"
        ][training_df_w_previous_games_average_features_w["game_nb"] == 6].values[0]

        before_average_lastfivegame_pts_tm_1 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lastfivegame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 6].values[0]
        )

        before_average_lasttengame_pts_tm_1 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lasttengame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 6].values[0]
        )

        before_average_game_pts_tm_2 = training_df_w_previous_games_average_features_w[
            "before_average_pts_tm"
        ][training_df_w_previous_games_average_features_w["game_nb"] == 8].values[0]

        before_average_lastfivegame_pts_tm_2 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lastfivegame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 8].values[0]
        )

        before_average_lasttengame_pts_tm_2 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lasttengame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 8].values[0]
        )

        before_average_game_pts_tm_3 = training_df_w_previous_games_average_features_w[
            "before_average_pts_tm"
        ][training_df_w_previous_games_average_features_w["game_nb"] == 34].values[0]

        before_average_lastfivegame_pts_tm_3 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lastfivegame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 34].values[
                0
            ]
        )

        before_average_lasttengame_pts_tm_3 = (
            training_df_w_previous_games_average_features_w[
                "before_average_lasttengame_pts_tm"
            ][training_df_w_previous_games_average_features_w["game_nb"] == 34].values[
                0
            ]
        )

        assert (
            before_average_game_pts_tm_1
            == before_average_lastfivegame_pts_tm_1
            == before_average_lasttengame_pts_tm_1
        )
        assert before_average_game_pts_tm_2 != before_average_lastfivegame_pts_tm_2
        assert (
            before_average_lastfivegame_pts_tm_2 == before_average_lasttengame_pts_tm_2
        )
        assert (
            before_average_lastfivegame_pts_tm_3
            != before_average_lasttengame_pts_tm_3
            != before_average_game_pts_tm_3
        )

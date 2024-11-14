from unittest import TestCase

import pandas as pd

from src.feature_engineering_functions.last_game_overtime import last_game_overtime


class TestLastGameOvertime(TestCase):
    def setUp(self) -> None:
        self.season = 2024
        self.team = "ATL"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_last_game_overtime(self):
        """
        GIVEN a valid nba game logs dataset
        WHEN the last_game_overtime function is called
        THEN the last_game_overtime feature should be equal to OT
        IF the previous game was with played with overtime
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(self.path_nba_gamelogs)

        nba_games_training_dataset_pre_cleaned = (
            nba_games_training_dataset_pre_cleaned.sort_values(
                ["id_season", "tm", "game_nb"]
            )
        )

        nba_games_w_last_game_overtime_features = last_game_overtime(
            nba_games_training_dataset_pre_cleaned
        )

        test_ot_data = nba_games_w_last_game_overtime_features[
            (nba_games_w_last_game_overtime_features["tm"] == self.team)
            & (nba_games_w_last_game_overtime_features["id_season"] == self.season)
        ]

        # Id of the game for where the team played OT
        ot_game = test_ot_data[test_ot_data["overtime"] == "OT"]["game_nb"].values[0]

        # game_nb where the overtime feature should be equal to OT
        game_nb_ot = ot_game + 1

        last_game_overtime_feature = test_ot_data[
            test_ot_data["game_nb"] == game_nb_ot
        ]["last_game_overtime"]

        assert last_game_overtime_feature.values[0] == "OT"

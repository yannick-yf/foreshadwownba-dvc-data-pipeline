from unittest import TestCase

import pandas as pd

from src.feature_engineering_functions.games_date_processing import (
    extract_days_of_week_from_date,
    game_on_weekend_features,
)


class TestLastGameDateProcessing(TestCase):
    def setUp(self) -> None:
        self.season = 2012
        self.team = "MIA"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )
        self.weekdays_reference = sorted(
            [
                "Sunday",
                "Tuesday",
                "Friday",
                "Wednesday",
                "Thursday",
                "Monday",
                "Saturday",
            ]
        )

    def test_extract_days_of_week_from_date(self):
        """
        GIVEN a sorted NBA game logs dataset
        WHEN extract_days_of_week_from_date is applied
        THEN each game should have the correct 'day_of_week' label
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(self.path_nba_gamelogs)

        nba_games_training_dataset_pre_cleaned = (
            nba_games_training_dataset_pre_cleaned.sort_values(
                ["id_season", "tm", "game_nb"]
            )
        )

        nba_games_w_days_of_week_from_date = extract_days_of_week_from_date(
            nba_games_training_dataset_pre_cleaned
        )

        nba_games_w_days_of_week_from_date = nba_games_w_days_of_week_from_date[
            (nba_games_w_days_of_week_from_date["tm"] == self.team)
            & (nba_games_w_days_of_week_from_date["id_season"] == self.season)
        ]

        # Id of the game for where the team played OT
        day_of_week_values = nba_games_w_days_of_week_from_date[
            "day_of_week"
        ].value_counts()
        week_days_list = sorted(list(day_of_week_values.index))

        assert (self.weekdays_reference == week_days_list) and (
            nba_games_w_days_of_week_from_date[
                nba_games_w_days_of_week_from_date["game_date"] == "2012-01-01"
            ]["day_of_week"].values[0]
            == "Sunday"
        )

    def test_game_on_weekend_features(self):
        """
        GIVEN a sorted NBA game logs dataset
        WHEN game_on_weekend_features is applied
        THEN games on weekends should be labeled as 'weekend'
        IF the game day is a weekend day, e.g., Sunday
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(self.path_nba_gamelogs)

        nba_games_training_dataset_pre_cleaned = (
            nba_games_training_dataset_pre_cleaned.sort_values(
                ["id_season", "tm", "game_nb"]
            )
        )

        nba_games_w_days_of_week_from_date_w_weekend = game_on_weekend_features(
            extract_days_of_week_from_date(nba_games_training_dataset_pre_cleaned)
        )

        assert (
            nba_games_w_days_of_week_from_date_w_weekend[
                nba_games_w_days_of_week_from_date_w_weekend["day_of_week"] == "Sunday"
            ]["week_weekend"].values[0]
            == "weekend"
        )

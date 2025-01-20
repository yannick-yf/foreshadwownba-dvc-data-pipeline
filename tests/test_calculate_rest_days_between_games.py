from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.rest_days_between_games import (
    calculate_rest_days_between_games,
)

class TestRestDaysBetweenGames(TestCase):
    def setUp(self) -> None:
        self.season = 2018
        self.team = "DAL"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_calculate_rest_days_between_games(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_rest_days_between_games is applied
        THEN it should correctly compute the number of rest days between two games
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        training_dataset_filtered = training_dataset[
            (training_dataset['tm']==self.team) &
            (training_dataset['id_season']==self.season)
            ]

        rest_days_features = calculate_rest_days_between_games(training_dataset_filtered)

        # Building ground truth value
        date_game_10 = rest_days_features['game_date'][rest_days_features['game_nb']==10].values[0]
        date_game_11 = rest_days_features['game_date'][rest_days_features['game_nb']==11].values[0]
        difference_in_days = (date_game_11 - date_game_10).astype('timedelta64[D]').astype(int)

        assert rest_days_features['rest'][rest_days_features['game_nb']==11].values[0] == difference_in_days
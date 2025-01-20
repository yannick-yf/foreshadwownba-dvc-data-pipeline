from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.last_games_average_features import (
    previous_season_ratio_features, 
    calculate_last_season_ratio, 
    calculate_max_game_id
)

class TestPreviousSeasonRatioFeatures(TestCase):
    def setUp(self) -> None:
        self.season = 2023
        self.team = "LAL"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_calculate_max_game_id(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_max_game_id function is applied
        THEN it should correctly compute the max number of games for a given team/season
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])
        
        training_dataset_filtered=training_dataset[training_dataset['tm']==self.team]

        max_game_id = calculate_max_game_id(training_dataset_filtered)

        assert max_game_id['max_game_id'][max_game_id['id_season']==self.season].values[0] == 82

    def test_calculate_last_season_ratio(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_last_season_ratio function is applied
        THEN it should correctly compute win ratio from last season
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])
        
        training_dataset_filtered = training_dataset[training_dataset['tm']==self.team]

        max_game_id = calculate_max_game_id(training_dataset_filtered)

        training_dataset_filtered = pd.merge(
            training_dataset_filtered, 
            max_game_id, 
            how="left", 
            on=["id_season", "tm"])

        training_dataset_filtered = calculate_last_season_ratio(training_dataset_filtered)

        assert training_dataset_filtered['before_season_ratio'][training_dataset_filtered['id_season_plus1']-1==self.season].values[0].round(5) == 0.52439

    def test_previous_season_ratio_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_last_season_ratio function is applied
        THEN it should correctly compute win ratio from last season
        """

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])
        
        training_dataset_filtered = training_dataset[training_dataset['tm']==self.team]

        last_season_ratio_features = previous_season_ratio_features(
            training_dataset_filtered
            )

        assert last_season_ratio_features['before_season_ratio'][last_season_ratio_features['id_season']==self.season+1].values[0].round(5) == 0.52439
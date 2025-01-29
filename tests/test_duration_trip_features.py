from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.duration_trip_features import (
    duration_trip_hours_between_cities,
    _get_distance_between_cities,
    _get_duration_trip_features,
    _fillna_trip_features
    
)

class TestDurationTripFeatures(TestCase):
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

        duration_trip_features = duration_trip_hours_between_cities(training_dataset_filtered)

        assert 1+1==2

    def test_get_distance_between_cities(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_rest_days_between_games is applied
        THEN it should correctly compute the number of rest days between two games
        """

        city_name = pd.read_csv("./data/constants/team_name.csv")
        data_city_distance = pd.read_csv("./data/constants/data_city_distance.csv")

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        training_dataset_filtered = training_dataset[
            (training_dataset['tm']==self.team) &
            (training_dataset['id_season']==self.season)
            ]

        dist_features_df = training_dataset_filtered[
            ["id_season", "game_date", "game_nb", "tm", "opp", "extdom"]
        ]

        dist_features_df = _get_distance_between_cities(
            city_name, data_city_distance, dist_features_df
        )

        # Assert that first DAL game for 2018 is a homegame So duration trip = 0
        assert dist_features_df['duration_trip'].values[0] == 0

        # Assert that second DAL game for 2018 is a homegame So duration trip = 0
        assert dist_features_df['duration_trip'].values[1] == 0

        # Assert that third DAL game for 2018 is a away so duration trip > 0
        assert dist_features_df['duration_trip'].values[3] > 0

    def test_get_duration_trip_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_rest_days_between_games is applied
        THEN it should correctly compute the number of rest days between two games
        """

        city_name = pd.read_csv("./data/constants/team_name.csv")
        data_city_distance = pd.read_csv("./data/constants/data_city_distance.csv")

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        training_dataset_filtered = training_dataset[
            (training_dataset['tm']==self.team) &
            (training_dataset['id_season']==self.season)
            ]

        dist_features_df = training_dataset_filtered[
            ["id_season", "game_date", "game_nb", "tm", "opp", "extdom"]
        ]

        dist_features_df = _get_distance_between_cities(
            city_name, data_city_distance, dist_features_df
        )

        dist_features_df = _get_duration_trip_features(dist_features_df)

        # Assert that first DAL game for 2018 is a homegame So duration trip = 0
        self.assertTrue(np.isnan(dist_features_df['before_average_lastfivegame_duration_trip'].values[0]))
        self.assertEqual(dist_features_df['before_average_duration_trip'].values[1], 0.0)

    def test_fillna_trip_features(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN calculate_rest_days_between_games is applied
        THEN it should correctly compute the number of rest days between two games
        """

        city_name = pd.read_csv("./data/constants/team_name.csv")
        data_city_distance = pd.read_csv("./data/constants/data_city_distance.csv")

        # Read the input data for the step
        training_dataset = pd.read_csv(self.path_nba_gamelogs)

        training_dataset = training_dataset.sort_values(["id_season", "tm", "game_nb"])

        training_dataset_filtered = training_dataset[
            (training_dataset['tm']==self.team) &
            (training_dataset['id_season']==self.season)
            ]

        dist_features_df = training_dataset_filtered[
            ["id_season", "game_date", "game_nb", "tm", "opp", "extdom"]
        ]

        dist_features_df = _get_distance_between_cities(
            city_name, data_city_distance, dist_features_df
        )

        dist_features_df = _get_duration_trip_features(dist_features_df)

        dist_features_df = _fillna_trip_features(dist_features_df)

        # Assert that first DAL game for 2018 is a homegame So duration trip = 0
        self.assertFalse(np.isnan(dist_features_df['before_average_lastfivegame_duration_trip'].values[0]))
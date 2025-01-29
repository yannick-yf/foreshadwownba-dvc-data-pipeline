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

    def test_get_distance_between_cities(self):
        """
        GIVEN a dataset of NBA game logs with team and season details,
        WHEN _get_distance_between_cities is applied,
        THEN it should:
            - Compute the correct distance traveled between consecutive games.
            - Ensure home games have zero travel distance.
            - Verify that away games have a positive travel distance.
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
        GIVEN a dataset of NBA game logs with team and season details,
        WHEN _get_duration_trip_features is applied,
        THEN it should:
            - Compute cumulative travel duration for previous games.
            - Ensure NaN values are correctly assigned where no previous data is available.
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
        GIVEN a dataset of NBA game logs with team and season details,
        WHEN _fillna_trip_features is applied,
        THEN it should:
            - Fill missing values for trip duration features.
            - Ensure no NaN values remain after application.
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
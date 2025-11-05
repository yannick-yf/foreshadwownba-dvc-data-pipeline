from unittest import TestCase

import pandas as pd
import numpy as np
from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features, previous_games_win_ratio_average_features,
)

from src.feature_engineering_functions.duration_trip_features import (
    duration_trip_hours_between_cities,
)

from src.feature_engineering_functions.rest_days_between_games import (
    calculate_rest_days_between_games,
)
from src.feature_engineering_functions.last_game_overtime import last_game_overtime
from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features,
    previous_games_win_ratio_average_features,
    previous_season_ratio_features,
)
from src.feature_engineering_functions.previous_days_average_features import _game_indicator_columns, _recode_game_day_trip_duration

class TestPreviousDaysAverageFeatures(TestCase):
    def setUp(self) -> None:
        self.season = 2023
        self.team = "BOS"
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_game_indicator_columns(self):
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

        training_dataset = (
            training_dataset[
                (training_dataset["tm"] == self.team)
                & (
                    training_dataset["id_season"]
                    == self.season
                )
            ]
        )

        training_df_processed = (training_dataset
            .pipe(previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"])
            .pipe(previous_games_win_ratio_average_features)
            .pipe(previous_season_ratio_features)
            .pipe(calculate_rest_days_between_games)
            .pipe(duration_trip_hours_between_cities)
        )

        # ------------------------------
        # STEP 1
        subset_1 = training_df_processed
        subset_1["Date"] = pd.to_datetime(subset_1["game_date"])

        # ------------------------------
        # STEP 2
        subset_2 = training_df_processed[["id_season", "game_date", "tm", "duration_trip"]]

        # ------------------------------
        # STEP 3
        subset_1_1 = subset_1.groupby(["id_season", "tm"]).apply(
            lambda x: x.drop_duplicates("Date").set_index("Date").resample("D").ffill()
        )

        # Rename multi index
        subset_1_1.index = subset_1_1.index.set_names(
            ["id_season_index", "tm_index", "date"]
        )
        subset_1_1 = subset_1_1.reset_index()

        subset_1_1 = subset_1_1.drop(columns=["id_season_index", "tm_index"])
        subset_1_1 = subset_1_1[
            ["date", "id_season", "game_nb", "game_date", "extdom", "tm", "opp"]
        ]

        last_days_features = pd.merge(
            subset_1_1,
            subset_2,
            how="left",
            left_on=["id_season", "game_date", "tm"],
            right_on=["id_season", "game_date", "tm"],
        )

        # ------------------------------
        # STEP 4

        last_days_features = _game_indicator_columns(last_days_features)

        ref_unique_values_dom_y_n = last_days_features[
            (last_days_features["date"] == last_days_features["game_date"]) &
            (last_days_features["extdom"] == "dom")
            ]['dom_y_n'].value_counts()

        # We assert that under the condition taht game is at the calendar date and game is dom then it is equal to 1
        assert ref_unique_values_dom_y_n.shape[0] == 1

    def test_recode_game_day_trip_duration(self):

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

        training_dataset = (
            training_dataset[
                (training_dataset["tm"] == self.team)
                & (
                    training_dataset["id_season"]
                    == self.season
                )
            ]
        )

        training_df_processed = (training_dataset
            .pipe(previous_games_average_features, columns_to_process=["pts_tm", "pts_opp"])
            .pipe(previous_games_win_ratio_average_features)
            .pipe(previous_season_ratio_features)
            .pipe(calculate_rest_days_between_games)
            .pipe(duration_trip_hours_between_cities)
        )

        # ------------------------------
        # STEP 1
        subset_1 = training_df_processed
        subset_1["Date"] = pd.to_datetime(subset_1["game_date"])

        # ------------------------------
        # STEP 2
        subset_2 = training_df_processed[["id_season", "game_date", "tm", "duration_trip"]]

        # ------------------------------
        # STEP 3
        subset_1_1 = subset_1.groupby(["id_season", "tm"]).apply(
            lambda x: x.drop_duplicates("Date").set_index("Date").resample("D").ffill()
        )

        # Rename multi index
        subset_1_1.index = subset_1_1.index.set_names(
            ["id_season_index", "tm_index", "date"]
        )
        subset_1_1 = subset_1_1.reset_index()

        subset_1_1 = subset_1_1.drop(columns=["id_season_index", "tm_index"])
        subset_1_1 = subset_1_1[
            ["date", "id_season", "game_nb", "game_date", "extdom", "tm", "opp"]
        ]

        last_days_features = pd.merge(
            subset_1_1,
            subset_2,
            how="left",
            left_on=["id_season", "game_date", "tm"],
            right_on=["id_season", "game_date", "tm"],
        )

        # ------------------------------
        # STEP 4

        last_days_features = _game_indicator_columns(last_days_features)

        last_days_features = _recode_game_day_trip_duration(last_days_features)

        values_to_assert = last_days_features["duration_trip_y_n"][
            (last_days_features["date"] != last_days_features["game_date"])
            ].sum()

        assert values_to_assert == 0
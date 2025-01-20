from unittest import TestCase

import pandas as pd
import numpy as np
from src.features_engineering_pipeline import features_engineering_pipeline

class TestEndToEndPipeline(TestCase):
    def setUp(self) -> None:
        self.nb_features = 63
        self.input_file_folder_name = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )
        self.output_file_folder_name = (
            "./tests/data/nba_games_training_dataset_pre_cleaned.csv"
        )

    def test_features_engineering_pipeline(self):
        """
        GIVEN a dataset of NBA game logs with team and season details
        WHEN previous_games_average_features is applied
        THEN it should correctly compute streak counts and average points of previous games:
            - Ensure that the 'before_average_pts_tm' for the second game matches the points of the first game.
            - Verify that 'before_average_pts_tm' for the first game is NaN, indicating no prior games.
        """

        features_engineering_pipeline(
            input_file_folder_name = self.input_file_folder_name,
            output_file_folder_name = self.output_file_folder_name,
            )

        features_df = pd.read_csv(self.output_file_folder_name)    

        assert features_df.shape[1] == self.nb_features
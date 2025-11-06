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
        GIVEN a pre-cleaned dataset of NBA game logs with relevant team and season details,
        WHEN the features_engineering_pipeline function is executed,
        THEN it should:
            - Generate a dataset with the expected number of features.
        """

        features_engineering_pipeline(
            input_file_folder_name = self.input_file_folder_name,
            output_file_folder_name = self.output_file_folder_name,
            )

        features_df = pd.read_csv(self.output_file_folder_name)    

        assert features_df.shape[1] == self.nb_features
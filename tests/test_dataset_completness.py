from unittest import TestCase

import pandas as pd


class TestDatasetCompleteness(TestCase):
    def setUp(self) -> None:
        self.path_nba_gamelogs = (
            "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        )
        self.expected_seasons = set(range(2009, 2027))

    def test_all_seasons_present_in_dataset(self):
        """
        GIVEN a processed NBA game logs dataset
        WHEN checking the available seasons
        THEN all seasons from 2009 to 2026 should be present
        """
        nba_games = pd.read_csv(self.path_nba_gamelogs)
        actual_seasons = set(nba_games["id_season"].unique())

        missing_seasons = self.expected_seasons - actual_seasons
        extra_seasons = actual_seasons - self.expected_seasons

        assert not missing_seasons, f"Missing seasons: {sorted(missing_seasons)}"
        # Optional: warn about unexpected extra seasons
        if extra_seasons:
            print(f"Note: Found additional seasons not in expected range: {sorted(extra_seasons)}")
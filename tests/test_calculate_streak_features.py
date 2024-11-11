from unittest import TestCase

import pandas as pd

from src.feature_engineering_functions.games_streak_features import calculate_streak_features

class TestLastGameOvertime(TestCase):
    def setUp(self) -> None:
        self.season = 2017
        self.team='GSW'
        self.path_nba_gamelogs = "./data/processed/nba_games_training_dataset_pre_cleaned.csv"

    def test_calculate_streak_features(self):
        """
        GIVEN a NBA game logs dataset
        WHEN calculate_streak_features is applied
        THEN streak counts should be correct
        e.g if streak_w_l is 'W 4' then before_streak_w_l should be 4
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(
            self.path_nba_gamelogs
            )
        
        nba_games_training_dataset_pre_cleaned = nba_games_training_dataset_pre_cleaned.sort_values(
            ['id_season','tm', 'game_nb']
        )

        nba_games_w_streak_games_features = calculate_streak_features(
            nba_games_training_dataset_pre_cleaned
            )
        
        test_streak_games_features = nba_games_w_streak_games_features[
            (nba_games_w_streak_games_features['tm']==self.team) &
            (nba_games_w_streak_games_features['id_season']==self.season)]

        # Id of the game for where the team played OT
        streak_w_l_game = test_streak_games_features[
            test_streak_games_features['streak_w_l']=='W 4'
            ]['game_nb'].values[0]

        # game_nb where the overtime feature should be equal to OT
        streak_w_l_game = streak_w_l_game + 1

        last_streak_w_l_game_feature = test_streak_games_features[
            test_streak_games_features['game_nb']==streak_w_l_game
            ]['before_streak_w_l']

        assert last_streak_w_l_game_feature.values[0] == 4

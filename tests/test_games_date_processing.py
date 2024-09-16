from unittest import TestCase

import pandas as pd

from src.feature_engineering_functions.games_date_processing import (
    extract_days_of_week_from_date,
    game_on_weekend_features,
)

class TestLastGameDateProcessing(TestCase):
    def setUp(self) -> None:
        self.season = 2024
        self.team='ATL'
        self.path_nba_gamelogs = "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
        self.weekdays_reference = sorted(['Sunday', 'Tuesday', 'Friday', 'Wednesday', 'Thursday', 'Monday', 'Saturday'])

    def test_extract_days_of_week_from_date(self):
        """
        GIVEN a valid nba game logs dataset
        WHEN the last_game_overtime function is called
        THEN the last_game_overtime feature should be equal to OT 
        IF the previous game was with played with overtime
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(
            self.path_nba_gamelogs
            )
        
        nba_games_training_dataset_pre_cleaned = nba_games_training_dataset_pre_cleaned.sort_values(
            ['id_season','tm', 'game_nb']
        )
        
        nba_games_w_days_of_week_from_date = extract_days_of_week_from_date(
            nba_games_training_dataset_pre_cleaned
            )
        
        nba_games_w_days_of_week_from_date = nba_games_w_days_of_week_from_date[
            (nba_games_w_days_of_week_from_date['tm']=='MIA') &
            (nba_games_w_days_of_week_from_date['id_season']==2012)]
        
        # Id of the game for where the team played OT
        day_of_week_values = nba_games_w_days_of_week_from_date['day_of_week'].value_counts()
        week_days_list = sorted(list(day_of_week_values.index))

        assert (self.weekdays_reference == week_days_list) and\
            (nba_games_w_days_of_week_from_date[
                nba_games_w_days_of_week_from_date['game_date']=='2012-01-01'
                ]['day_of_week'].values[0] == 'Sunday')

    def test_game_on_weekend_features(self):
        """
        GIVEN a valid nba game logs dataset
        WHEN the last_game_overtime function is called
        THEN the last_game_overtime feature should be equal to OT 
        IF the previous game was with played with overtime
        """

        nba_games_training_dataset_pre_cleaned = pd.read_csv(
            self.path_nba_gamelogs
            )
        
        nba_games_training_dataset_pre_cleaned = nba_games_training_dataset_pre_cleaned.sort_values(
            ['id_season','tm', 'game_nb']
        )
        
        nba_games_w_days_of_week_from_date_w_weekend = game_on_weekend_features(
            extract_days_of_week_from_date(nba_games_training_dataset_pre_cleaned)
            )

        assert nba_games_w_days_of_week_from_date_w_weekend[
                nba_games_w_days_of_week_from_date_w_weekend['day_of_week']=='Sunday'
                ]['week_weekend'].values[0] == 'weekend'
        
"""
This module performs pre-cleaning on the NBA games training dataset.
"""


from pandantic import BaseModel
from pydantic.types import StrictInt
from pydantic import Field, field_validator

import argparse
from pathlib import Path
import pandas as pd
import yaml
import os

from src.utils.logs import get_logger

logger = get_logger(
    "TRAINING DATASET ASSESSMENT", log_level='INFO'
)

team_list = [
    'ATL',
    'MEM',
    'PHO',
    'CHI',
    'CLE',
    'DAL',
    'UTA',
    'DET',
    'GSW',
    'TOR',
    'POR',
    'ORL',
    'MIL',
    'NYK',
    'WAS',
    'SAC',
    'MIN',
    'DEN',
    'OKC',
    'LAC',
    'IND',
    'SAS',
    'PHI',
    'MIA',
    'HOU',
    'BOS',
    'LAL',
    'BRK',
    'NOP',
    'CHO',
    'CHA',
    'NOH',
    'NJN']

extdom_list = ['ext', 'dom']

class DataFrameSchema(BaseModel):
    """Example schema for testing."""
    
    id_season: StrictInt = Field(ge=2000, le=2025)
    tm: str = Field(strict=True, min_length=3, max_length=3)
    opp: str = Field(strict=True, min_length=3, max_length=3)
    before_average_pts_tm: float = Field(ge=30, le=200)
    game_nb: StrictInt = Field(ge=2, le=82)
    extdom: str = Field(strict=True, min_length=3, max_length=3)
    results: StrictInt = Field(ge=0, le=1)
    pts_tm: StrictInt = Field(ge=0, le=250)
    pts_opp: StrictInt = Field(ge=0, le=250)
    before_average_lastfivegame_pts_tm: float = Field(ge=0.0, le=200.0)
    before_average_lasttengame_pts_tm: float = Field(ge=0.0, le=200.0)
    diff_all_minus_lastfivegame_pts_tm: float = Field(ge=-200.0, le=200.0)
    diff_all_minus_lasttengame_pts_tm: float = Field(ge=-200.0, le=200.0)
    before_average_pts_opp: float = Field(ge=0.0, le=200.0)
    before_average_lastfivegame_pts_opp: float = Field(ge=0.0, le=200.0)
    before_average_lasttengame_pts_opp: float = Field(ge=0.0, le=200.0)
    diff_all_minus_lastfivegame_pts_opp: float = Field(ge=-200.0, le=200.0)
    diff_all_minus_lasttengame_pts_opp: float = Field(ge=-200.0, le=200.0)
    before_average_W_ratio: float = Field(ge=0.0, le=1.0)
    before_average_lastfivegame_W_ratio: float = Field(ge=0.0, le=1.0)
    before_average_lasttengame_W_ratio: float = Field(ge=0.0, le=1.0)
    diff_all_minus_lastfivegame_W_ratio: float = Field(ge=-1.0, le=1.0)
    diff_all_minus_lasttengame_W_ratio: float = Field(ge=-1.0, le=1.0)
    before_season_ratio: float = Field(ge=0.0, le=1.0)
    rest:  StrictInt = Field(ge=0)
    duration_trip: float = Field(ge=0.0)
    before_average_duration_trip: float = Field(ge=0.0)
    before_average_lastfivegame_duration_trip: float = Field(ge=0.0)
    before_average_lasttengame_duration_trip: float = Field(ge=0.0)
    before_sum_lastfivegame_duration_trip: float = Field(ge=0.0)
    before_sum_lasttengame_duration_trip: float = Field(ge=0.0)
    nb_games_last_5days: float = Field(ge=0.0)
    nb_games_last_7days: float = Field(ge=0.0)
    nb_games_last_10days: float = Field(ge=0.0)
    nb_ext_games_last_5days: float = Field(ge=0.0)
    nb_ext_games_last_7days: float = Field(ge=0.0)
    nb_ext_games_last_10days: float = Field(ge=0.0)
    nb_dom_games_last_5days: float = Field(ge=0.0)
    nb_dom_games_last_7days: float = Field(ge=0.0)
    nb_dom_games_last_10days: float = Field(ge=0.0)
    sum_duration_trip_opp_n_last_5days: float = Field(ge=0.0)
    sum_duration_trip_opp_n_last_7days: float = Field(ge=0.0)
    sum_duration_trip_opp_n_last_10days: float = Field(ge=0.0, le=150.4)
    week_weekend: str
    before_ratio_win_ext: float = Field(ge=0.0, le=1.0)
    before_ratio_win_dom: float = Field(ge=0.0, le=1.0)
    last_game_overtime: str
    before_streak_w_l: float = Field(ge=-81, le=81)
    extdom_ext: StrictInt = Field(ge=0, le=1)
    last_game_overtime_Multiple_OT: StrictInt = Field(ge=0, le=1)
    last_game_overtime_NOT: StrictInt = Field(ge=0, le=1)
    last_game_overtime_OT: StrictInt = Field(ge=0, le=1)
    last_game_overtime_nan: StrictInt = Field(ge=0, le=0)
    tm_opp: str
    opp_opp: str
    before_average_W_ratio_opp: float = Field(ge=0.0, le=1.0)

    @field_validator('*', mode='before')
    def validate_non_null(cls, value):
        if value is None:
            raise ValueError("Field cannot be null")
        return value

    @field_validator("tm", "opp", "tm_opp", "opp_opp")
    @classmethod
    def validate_team(cls, v):
        if v not in team_list:
            raise ValueError("Value: " + v + ' is not accepted. Accepted value are: ' + str(team_list))
        
    @field_validator("extdom")
    @classmethod
    def validate_extdom(cls, v):
        if v not in extdom_list:
            raise ValueError("Value: " + v + ' is not accepted. Accepted value are: ' + str(extdom_list))
        
    # @field_validator("example_int")
    # def validate_even_integer(
    #     cls, x: int
    # ) -> int:
    #     """Example custom validator to validate if int is even."""
    #     if x % 2 != 0:
    #         raise ValidationError(f"example_int must be even, is {x}.")
    #     return x


#     @field_validator('*', mode='before')
#     def validate_non_null(cls, value):
#         if value is None:
#             raise ValueError("Field cannot be null")
#         return value

# class DataFrameListModel(RootModel):
#     root: List[DataFrameSchema]

# def validate_dataframe(df: pd.DataFrame) -> bool:
#     records = df.to_dict(orient='records')
#     try:
#         # Wrap records in the list-based root model
#         DataFrameListModel.model_validate(records)
#         return True
#     except ValidationError as e:
#         print(e)
#         return False

def training_dataset_assessment(config_path: Path) -> pd.DataFrame:
    """
    """
    with open(config_path, encoding="utf-8") as conf_file:
        config_params = yaml.safe_load(conf_file)

    nba_games_training_dataset = pd.read_csv(
        "./data/output/nba_games_training_dataset_final.csv"
    )
    logger.info("Shape of the DataFrame %s", str(nba_games_training_dataset.shape))

    # With the fulll dataframe
    df_raised_error = DataFrameSchema.parse_df(
        dataframe=nba_games_training_dataset,
        errors="raise",
    )

    logger.info("Training dataset assessment step complete")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config-params", dest="config_params", required=True)
    args = arg_parser.parse_args()
    training_dataset_assessment(config_path=args.config_params)
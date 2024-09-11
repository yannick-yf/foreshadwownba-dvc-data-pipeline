from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import pandas as pd
import numpy as np
from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger

from src.feature_engineering_functions.last_games_average_features import (
    previous_games_average_features,
    previous_games_ratio_average_features,
    previous_season_ratio_features,
)
from src.feature_engineering_functions.rest_days_between_games import (
    rest_days_between_games,
)
from src.feature_engineering_functions.duration_trip_features import (
    duration_trip_hours_between_cities,
)
from src.feature_engineering_functions.previous_days_average_features import (
    previous_days_average_features,
)
from src.feature_engineering_functions.games_date_processing import (
    game_on_weekend_features,
    exctract_days_of_week_from_date,
)
from src.feature_engineering_functions.average_ratio_win_loose_ext_game import (
    average_ratio_win_loose_ext_game,
)
from src.feature_engineering_functions.last_game_overtime import last_game_overtime
from src.feature_engineering_functions.streack_w_l import streack_w_l
from src.feature_engineering_functions.final_cleaning import final_cleaning


def copy_df(df):
    return df.copy()


def features_engineering_pipeline(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "LAST_GAMES_RATIO_AVERAGE_FEATURES",
        log_level=config_params["base"]["log_level"],
    )

    # Read the input data for the step
    training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
    )

    training_dataset_w_features = (
        training_dataset.pipe(copy_df)
        .pipe(previous_games_average_features)
        .pipe(previous_games_ratio_average_features)
        .pipe(previous_season_ratio_features)
        .pipe(rest_days_between_games)
        .pipe(duration_trip_hours_between_cities)
        .pipe(previous_days_average_features)
        .pipe(exctract_days_of_week_from_date)
        .pipe(game_on_weekend_features)
        .pipe(average_ratio_win_loose_ext_game)
        .pipe(last_game_overtime)
        .pipe(streack_w_l)
        .pipe(final_cleaning)
    )

    # Save the data
    logger.info(
        "Final Shape of the Dataframe after featuer engineering process "
        + str(training_dataset_w_features.shape)
    )
    training_dataset_w_features.to_csv(
        "./data/processed/nba_games_training_dataset_cleaned_w_features.csv",
        index=False,
    )

    logger.info("Feateure Engineering Generation step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    features_engineering_pipeline(config_path=args.config_params)


def FeaturesEngineeringPipeline(TRAINING_DF):

    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Last games average features
    TRAINING_DF = LastGamesAverageFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # #----------------------------------------------------
    # # Clustering method to classify type of teams based on style of play
    # MaxGameID = pd.DataFrame(TRAINING_DF.groupby(['id_season','tm'])['game_nb'].max()).reset_index()
    # MaxGameID.columns = ['id_season', 'tm', 'max_game_id']

    # test =  pd.merge(
    #     TRAINING_DF,
    #     MaxGameID,
    #     how='left',
    #     left_on=['id_season', 'tm'],
    #     right_on=['id_season','tm'])

    # test = test[test['game_nb']==test['max_game_id']]

    # test = test[['id_season', 'tm',
    #     'before_average_pts_tm', 'before_average_pts_opp', 'before_average_fg_tm', 'before_average_fga_tm', 'before_average_fg_prct_tm', 'before_average_3p_tm', 'before_average_3pa_tm',
    #     'before_average_3p_prct_tm', 'before_average_ft_tm', 'before_average_fta_tm', 'before_average_ft_prct_tm', 'before_average_orb_tm', 'before_average_trb_tm',
    #     'before_average_ast_tm', 'before_average_stl_tm', 'before_average_blk_tm', 'before_average_tov_tm', 'before_average_pf_tm', 'before_average_fg_opp', 'before_average_fga_opp',
    #     'before_average_fg_prct_opp', 'before_average_3p_opp', 'before_average_3pa_opp', 'before_average_3p_prct_opp', 'before_average_ft_opp', 'before_average_fta_opp',
    #     'before_average_ft_prct_opp', 'before_average_orb_opp', 'before_average_trb_opp', 'before_average_ast_opp', 'before_average_stl_opp', 'before_average_blk_opp',
    #     'before_average_tov_opp', 'before_average_pf_opp']]

    # print(test.head())
    # sys.exit()

    # -------------------------------------------
    # EloFeatures
    TRAINING_DF = EloFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Last games ratio average features
    TRAINING_DF = LastGamesRatioAverageFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Last games ratio average features
    TRAINING_DF = LastSeasonRatioFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Rest days between two games
    TRAINING_DF = RestDaysBetweenGames(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Trip Duration between cities
    TRAINING_DF = DurationTripHoursBetweenCities(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Lsst Xdays average features
    TRAINING_DF = LastDaysAverageFeatures(TRAINING_DF)  # ADD 12 variables
    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # Day of Week end Game on Weekend
    TRAINING_DF = ExctractDaysofWeekFromDate(TRAINING_DF)
    TRAINING_DF = GameOnWeekendFeatures(TRAINING_DF)

    # ------------------------------------------------------------------
    # Average ration team against good/bad team - TODO
    # TRAINING_DF = AverageRatioVsGoodBadTeam(TRAINING_DF)

    # ------------------------------------------------------------------
    # Average ratio team at the ext dom - TODO
    TRAINING_DF = AverageRatioWLExtGame(TRAINING_DF)

    # -------------------------------------------
    # Overtime Last Games
    TRAINING_DF["last_game_overtime"] = TRAINING_DF.groupby(["id_season", "tm"])[
        "overtime"
    ].shift(1)

    # -------------------------------------------
    # Streack W/L
    TRAINING_DF["streak_w_l_2"] = (
        TRAINING_DF["streak_w_l"].str.extract("(\d+)").astype(int)
    )
    TRAINING_DF["streak_w_l_2"] = np.where(
        TRAINING_DF["streak_w_l"].str.slice(0, 1) == "L",
        TRAINING_DF["streak_w_l_2"] * -1,
        TRAINING_DF["streak_w_l_2"],
    )

    TRAINING_DF["before_streak_w_l"] = TRAINING_DF.groupby(["id_season", "tm"])[
        "streak_w_l_2"
    ].shift(1)

    # -------------------------------------------
    # Remove first season we have
    TRAINING_DF = TRAINING_DF[TRAINING_DF["id_season"] > TRAINING_DF["id_season"].min()]

    # -------------------------------------------
    # Remove non needed features for the model
    TRAINING_DF = TRAINING_DF.drop(
        [
            #'pts_tm', 'pts_opp',
            "fg_tm",
            "fga_tm",
            "fg_prct_tm",
            "3p_tm",
            "3pa_tm",
            "3p_prct_tm",
            "ft_tm",
            "fta_tm",
            "ft_prct_tm",
            "orb_tm",
            "trb_tm",
            "ast_tm",
            "stl_tm",
            "blk_tm",
            "tov_tm",
            "pf_tm",
            "fg_opp",
            "fga_opp",
            "fg_prct_opp",
            "3p_opp",
            "3pa_opp",
            "3p_prct_opp",
            "ft_opp",
            "fta_opp",
            "ft_prct_opp",
            "orb_opp",
            "trb_opp",
            "ast_opp",
            "stl_opp",
            "blk_opp",
            "tov_opp",
            "pf_opp",
            "w_tot",
            "l_tot",
            "streak_w_l",
            "streak_w_l_2",
            "overtime",
        ],
        axis=1,
    )

    # -------------------------------------------
    # Player Features: Injury and size - salary ect
    TRAINING_DF = PlayersFeatures(TRAINING_DF)

    # ------------------------------------------------------------------
    # Add opponent features
    # Get training dataset processed to get one row per games
    TRAINING_DF = fn_merge_to_opponents_data(TRAINING_DF)

    TRAINING_DF.rename(
        {"week_weekend_x": "week_weekend", "day_of_week_x": "day_of_week"},
        axis=1,
        inplace=True,
    )

    # ------------------------------------------------------------------
    # Add opponent features
    # Get training dataset processed to get one row per games
    TRAINING_DF = TmMinusOppFeatures(TRAINING_DF)

    # -------------------------------------------
    # Remove non needed features for the model
    TRAINING_DF = TRAINING_DF.drop(["week_weekend_y", "day_of_week_y"], axis=1)

    print(TRAINING_DF.shape[1])

    # -------------------------------------------
    # TODO: For all the variables : tm - opp OR best vs worst exemple before_average_lastfivegame_3p_tm_x - before_average_lastfivegame_3p_tm_y -
    # TODO: StreakW, StreakL, StreakEXT, StreakDOM - CANCEL FOR NOW
    # TODO: Average features home/awaygames - create for pts a new column where for ext game pts is nas then apply rolling means on this two columns. populate missing value using fillna backfill method
    # TODO: Averege features versu best teams

    return TRAINING_DF

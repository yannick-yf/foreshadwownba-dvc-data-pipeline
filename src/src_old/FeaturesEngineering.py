
import sys
import numpy as np
import pandas as pd
sys.path.append(".")

from functions.FeaturesEngineeringFunctions.LastGamesAverageFeatures import LastGamesAverageFeatures, LastGamesRatioAverageFeatures, LastSeasonRatioFeatures
from functions.FeaturesEngineeringFunctions.LastDaysAverageFeatures import LastDaysAverageFeatures
from functions.FeaturesEngineeringFunctions.RestDaysBetweenGames import RestDaysBetweenGames
from functions.FeaturesEngineeringFunctions.DurationTripFeatures import DurationTripHoursBetweenCities
from functions.FeaturesEngineeringFunctions.GameDateProcessing import ExctractDaysofWeekFromDate, GameOnWeekendFeatures
from functions.FeaturesEngineeringFunctions.EloFeatures import EloFeatures
from functions.fn_merge_to_opponents_data import fn_merge_to_opponents_data
from functions.FeaturesEngineeringFunctions.TmMinusOppFeatures import TmMinusOppFeatures
from functions.FeaturesEngineeringFunctions.AverageRatioWLExtGame import AverageRatioWLExtGame
from functions.FeaturesEngineeringFunctions.PlayersFeatures import PlayersFeatures

# https://towardsdatascience.com/using-pandas-pipe-function-to-improve-code-readability-96d66abfaf8

def FeaturesEngineeringPipeline(TRAINING_DF):

    print(TRAINING_DF.shape[1])

    #-------------------------------------------
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

    #-------------------------------------------
    # EloFeatures
    TRAINING_DF = EloFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Last games ratio average features
    TRAINING_DF = LastGamesRatioAverageFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Last games ratio average features
    TRAINING_DF = LastSeasonRatioFeatures(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Rest days between two games
    TRAINING_DF = RestDaysBetweenGames(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Trip Duration between cities
    TRAINING_DF = DurationTripHoursBetweenCities(TRAINING_DF)
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Lsst Xdays average features
    TRAINING_DF = LastDaysAverageFeatures(TRAINING_DF) # ADD 12 variables
    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # Day of Week end Game on Weekend
    TRAINING_DF = ExctractDaysofWeekFromDate(TRAINING_DF)
    TRAINING_DF = GameOnWeekendFeatures(TRAINING_DF)

    #------------------------------------------------------------------
    # Average ration team against good/bad team - TODO
    # TRAINING_DF = AverageRatioVsGoodBadTeam(TRAINING_DF)

    #------------------------------------------------------------------
    # Average ratio team at the ext dom - TODO
    TRAINING_DF = AverageRatioWLExtGame(TRAINING_DF)

    #-------------------------------------------
    # Overtime Last Games
    TRAINING_DF['last_game_overtime'] = TRAINING_DF.groupby(['id_season', 'tm'])['overtime'].shift(1)

    #-------------------------------------------
    # Streack W/L
    TRAINING_DF['streak_w_l_2'] = TRAINING_DF['streak_w_l'].str.extract('(\d+)').astype(int)
    TRAINING_DF['streak_w_l_2'] = np.where(
        TRAINING_DF['streak_w_l'].str.slice(0,1) == 'L',
        TRAINING_DF['streak_w_l_2']*-1,
        TRAINING_DF['streak_w_l_2']) 

    TRAINING_DF['before_streak_w_l'] = TRAINING_DF.groupby(['id_season', 'tm'])['streak_w_l_2'].shift(1)

    #-------------------------------------------
    # Remove first season we have 
    TRAINING_DF = TRAINING_DF[TRAINING_DF['id_season'] > TRAINING_DF['id_season'].min()]

    #-------------------------------------------
    # Remove non needed features for the model
    TRAINING_DF = TRAINING_DF.drop([
        #'pts_tm', 'pts_opp',
        'fg_tm', 'fga_tm', 'fg_prct_tm', '3p_tm', '3pa_tm',
        '3p_prct_tm', 'ft_tm', 'fta_tm', 'ft_prct_tm', 'orb_tm', 'trb_tm',
        'ast_tm', 'stl_tm', 'blk_tm', 'tov_tm', 'pf_tm', 'fg_opp', 'fga_opp',
        'fg_prct_opp', '3p_opp', '3pa_opp', '3p_prct_opp', 'ft_opp', 'fta_opp',
        'ft_prct_opp', 'orb_opp', 'trb_opp', 'ast_opp', 'stl_opp', 'blk_opp',
        'tov_opp', 'pf_opp', 'w_tot', 'l_tot', 'streak_w_l', 'streak_w_l_2', 'overtime'
        ], axis = 1)

    #-------------------------------------------
    # Player Features: Injury and size - salary ect
    TRAINING_DF = PlayersFeatures(TRAINING_DF)

    #------------------------------------------------------------------
    # Add opponent features
    # Get training dataset processed to get one row per games 
    TRAINING_DF = fn_merge_to_opponents_data(TRAINING_DF)

    TRAINING_DF.rename({
        'week_weekend_x': 'week_weekend',
        'day_of_week_x': 'day_of_week'}, axis=1, inplace=True)

    #------------------------------------------------------------------
    # Add opponent features
    # Get training dataset processed to get one row per games 
    TRAINING_DF = TmMinusOppFeatures(TRAINING_DF)

    #-------------------------------------------
    # Remove non needed features for the model
    TRAINING_DF = TRAINING_DF.drop([
        'week_weekend_y', 'day_of_week_y'
        ], axis = 1)

    print(TRAINING_DF.shape[1])

    #-------------------------------------------
    # TODO: For all the variables : tm - opp OR best vs worst exemple before_average_lastfivegame_3p_tm_x - before_average_lastfivegame_3p_tm_y - 
    # TODO: StreakW, StreakL, StreakEXT, StreakDOM - CANCEL FOR NOW
    # TODO: Average features home/awaygames - create for pts a new column where for ext game pts is nas then apply rolling means on this two columns. populate missing value using fillna backfill method
    # TODO: Averege features versu best teams
    
    return TRAINING_DF
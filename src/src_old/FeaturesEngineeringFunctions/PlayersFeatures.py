import pandas as pd
import numpy as np
import datetime
import sys

from functions.FeaturesEngineeringFunctions.PlayerFeaturesFunctions import PlayerInjuryTeamLevel, PlayerStatsAtGameLevel
pd.options.mode.chained_assignment = None

def PlayersFeatures(TRAINING_DF):

    PlayerStatsAtGameLevelFeatures = PlayerStatsAtGameLevel.PlayerStatsAtGameLevel()

    TRAINING_DF = pd.merge(
        TRAINING_DF,
        PlayerStatsAtGameLevelFeatures,
        left_on=['id_season', 'tm', 'game_date'],
        right_on=['id_season', 'tm', 'game_date'],
        how= 'left')

    PlayerInjuryTeamLevelFeatures = PlayerInjuryTeamLevel.PlayerInjuryTeamLevel()

    TRAINING_DF = pd.merge(
        TRAINING_DF,
        PlayerInjuryTeamLevelFeatures,
        left_on=['tm',  'game_date'],
        right_on=[ 'tm',  'game_date'],
        how= 'left')

    return TRAINING_DF

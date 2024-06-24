


import pandas as pd
import numpy as np
import datetime
import sys
pd.options.mode.chained_assignment = None

def PlayerInjuryTeamLevel():

    #------------------------------
    # STEP 1: Load Required Data

    PlayerBoxscores = pd.read_csv('foreshadwownba-data-engineering-pipeline-output/nba_player_boxscores_final_2022-09-16.csv')
    PlayerInjury = pd.read_csv('foreshadwownba-data-engineering-pipeline-output/nba_players_injury_data_2022-09-16.csv')

    #------------------------------------------------
    # convert minutes to numeric
    PlayerBoxscores['mp'] = pd.to_numeric(PlayerBoxscores['mp'].str.replace(':', '.'))
    PlayerBoxscores = PlayerBoxscores[PlayerBoxscores['mp'] >= 8]

    #------------------------------------------------
    # Stadardize Salary per season

    PlayerBoxscores['salary'] = round(PlayerBoxscores['salary'] / 1000000, 3)
    g = PlayerBoxscores.groupby(['id_season'])['salary']
    min_, max_ = g.transform('min'), g.transform('max')
    PlayerBoxscores['salary_scale'] = (PlayerBoxscores['salary'] - min_) / (max_ - min_)

    #------------------------------------------------
    # Averaege player stats per games shiffted

    PlayerBoxscores = PlayerBoxscores[[
        'id_season', 'tm','player_name', 'game_date',
        'mp',	'pts',	'ast',	'trb',	'player_game_score','plus_minus']].sort_values(by = ['player_name', 'id_season', 'tm', 'game_date'])

    colums_to_process = [
        'mp',	
        'pts',	
        'ast',	
        'trb',	
        'player_game_score',	
        'plus_minus'
        ]

    for col in colums_to_process:
        PlayerBoxscores['before_average_' + col ] = round(PlayerBoxscores.groupby(['id_season', 'player_name'])[col].transform(lambda x: x.shift(1).expanding().mean()), 1)

    PlayerBoxscoresFeatures = PlayerBoxscores[[
        'id_season', 'tm','player_name', 'game_date',
        'before_average_mp',	'before_average_pts',	'before_average_ast',	'before_average_trb',	'before_average_player_game_score',	'before_average_plus_minus']
        ]

    PlayerBoxscoresFeatures["MinutesRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_mp"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["PtsRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_pts"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["GameScoreRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_player_game_score"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["PlusMinusRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_plus_minus"].rank("dense", ascending=False)

    #-------------------------------------------------------------------------------------------------
    # Injury Features

    PlayerInjuryFiltered = PlayerInjury[['id_season',	'tm',	'game_date',	'player_name']]

    PlayerInjuryFilteredCompleted = pd.merge(
        PlayerInjuryFiltered,
        PlayerBoxscoresFeatures,
        left_on=['id_season', 'tm', 'player_name'],
        right_on=['id_season', 'tm', 'player_name'])

    PlayerInjuryFilteredCompleted['time_diff'] = pd.to_datetime(PlayerInjuryFilteredCompleted['game_date_x']) - pd.to_datetime(PlayerInjuryFilteredCompleted['game_date_y'])
    PlayerInjuryFilteredCompleted['time_diff'] = PlayerInjuryFilteredCompleted['time_diff'].dt.days
    PlayerInjuryFilteredCompleted = PlayerInjuryFilteredCompleted[PlayerInjuryFilteredCompleted['time_diff'] >= 0]

    PlayerInjuryFilteredCompleted = PlayerInjuryFilteredCompleted[PlayerInjuryFilteredCompleted['time_diff'] == PlayerInjuryFilteredCompleted.groupby(['id_season', 'tm', 'player_name', 'game_date_x'])['time_diff'].transform('min')]
    PlayerInjuryFilteredCompleted = PlayerInjuryFilteredCompleted.sort_values(['id_season', 'tm', 'player_name', 'game_date_x','time_diff']).drop_duplicates(subset=['id_season', 'tm', 'player_name', 'game_date_x'])

    #---------------------------
    # Aggreagtion at the team - game level 

    NumberPlayerInjured = PlayerInjuryFilteredCompleted.groupby(['tm', 'game_date_x']).size().reset_index()
    NumberPlayerInjured.columns = ['tm', 'game_date', 'nb_player_injured']
    NumberPlayerInjured[NumberPlayerInjured['nb_player_injured'] == NumberPlayerInjured['nb_player_injured'].max()].head()
        
    #---------------------------
    # Max ranking for pts/min etc describing the best players injured

    BestRankPlayerInjured = PlayerInjuryFilteredCompleted.groupby(['tm', 'game_date_x'])[[
        'PtsRanking',
        'PlusMinusRanking', 
        'MinutesRanking', 
        'GameScoreRanking']].min().reset_index()

    BestRankPlayerInjured.columns = [
        'tm', 
        'game_date', 
        'RankingBestPlayerInjured_Pts', 
        'RankingBestPlayerInjured_PlusMinus',
        'RankingBestPlayerInjured_Minutes',
        'RankingBestPlayerInjured_GameScores']
    
    BestValuesPlayerInjured = PlayerInjuryFilteredCompleted.groupby(['tm', 'game_date_x'])[['before_average_pts','before_average_plus_minus', 'before_average_mp', 'before_average_player_game_score']].max().reset_index()
    BestValuesPlayerInjured.columns = ['tm', 'game_date', 'AverageBestPlayerInjured_Pts', 'AverageBestPlayerInjured_PlusMinus','AverageBestPlayerInjured_Minutes','AverageBestPlayerInjured_GameScores']

    #-------------------------------
    # Get the time dif of the player with the best before_average_pts

    DurationBestPlayerInjured = PlayerInjuryFilteredCompleted.groupby(['tm', 'game_date_x','time_diff'])['before_average_pts'].max().reset_index()
    DurationBestPlayerInjured = DurationBestPlayerInjured.sort_values(by=['tm', 'game_date_x', 'before_average_pts'], ascending=False).drop_duplicates(subset=['tm', 'game_date_x'], keep='first')
    DurationBestPlayerInjured.columns = ['tm', 'game_date', 'DurationBestPlayerInjured_Pts', 'before_average_pts']
    DurationBestPlayerInjured = DurationBestPlayerInjured[['tm', 'game_date', 'DurationBestPlayerInjured_Pts']]

    Schedule = PlayerBoxscoresFeatures[[ 'tm', 'game_date']].drop_duplicates()

    PlayerInjuryTeamLevelFeatures = pd.merge(
        Schedule,
        NumberPlayerInjured,

        left_on=[ 'tm', 'game_date'],
        right_on=['tm', 'game_date'],
        how= 'left')

    PlayerInjuryTeamLevelFeatures = pd.merge(
        PlayerInjuryTeamLevelFeatures,
        DurationBestPlayerInjured,
        left_on=[ 'tm', 'game_date'],
        right_on=[ 'tm', 'game_date'],
        how= 'left')

    PlayerInjuryTeamLevelFeatures = pd.merge(
        PlayerInjuryTeamLevelFeatures,
        BestValuesPlayerInjured,
        left_on=[ 'tm', 'game_date'],
        right_on=[ 'tm', 'game_date'],
        how= 'left')

    PlayerInjuryTeamLevelFeatures = pd.merge(
        PlayerInjuryTeamLevelFeatures,
        BestRankPlayerInjured,
        left_on=['tm', 'game_date'],
        right_on=['tm', 'game_date'],
        how= 'left')

    PlayerInjuryTeamLevelFeatures = PlayerInjuryTeamLevelFeatures.fillna(0)

    PlayerInjuryTeamLevelFeatures['game_date'] = pd.to_datetime(PlayerInjuryTeamLevelFeatures['game_date'])
    
    return PlayerInjuryTeamLevelFeatures




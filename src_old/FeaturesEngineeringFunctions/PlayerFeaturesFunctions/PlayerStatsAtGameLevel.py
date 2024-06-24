


import pandas as pd
import numpy as np
import datetime
import sys
pd.options.mode.chained_assignment = None

def PlayerStatsAtGameLevel():

    #------------------------------
    # STEP 1: Load Required Data

    PlayerBoxscores = pd.read_csv('foreshadwownba-data-engineering-pipeline-output/nba_player_boxscores_final_2022-09-16.csv')

    #------------------------------------------------
    # STEP 2: Remove Specific player that didn't play a game
    PlayerBoxscores['mp'] = pd.to_numeric(PlayerBoxscores['mp'].str.replace(':', '.'))
    PlayerBoxscores = PlayerBoxscores[PlayerBoxscores['mp'] >= 8]

    #------------------------------------------------
    # STEP 3: Stadardize Salary per season

    PlayerBoxscores['salary'] = round(PlayerBoxscores['salary'] / 1000000, 3)
    g = PlayerBoxscores.groupby(['id_season'])['salary']
    min_, max_ = g.transform('min'), g.transform('max')
    PlayerBoxscores['salary_scale'] = (PlayerBoxscores['salary'] - min_) / (max_ - min_)

    #----------------------------------------------
    # STEP 4: Max salary per position  - Avg Salary per positon - Number player per position

    AvgPlayerAttributesPerPositionPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season','tm', 'game_date', 'Position'])[['Wt',	'Experience',	'Age',	'cm_size',	'salary', 'salary_scale']].mean()
        ).reset_index()

    NumberPlyerPerPositionPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season', 'tm', 'game_date', 'Position']).size().unstack('Position', fill_value=0)
        ).reset_index()

    NumberPlyerPerPositionPerGames.columns = ['id_season', 'tm', 'game_date', 'nb_C', 'nb_PF', 'nb_PG', 'nb_SF', 'nb_SG']

    MaxSalaryPerPositionPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season','tm', 'game_date', 'Position'])['salary_scale'].max()
        ).reset_index()

    MaxSalaryPerPositionPerGames = MaxSalaryPerPositionPerGames.pivot(index=['id_season', 'tm', 'game_date'], columns='Position', values='salary_scale').reset_index()
    MaxSalaryPerPositionPerGames.columns = ['id_season', 'tm', 'game_date', 'max_salary_C', 'max_salary_PF', 'max_salary_PG', 'max_salary_SF', 'max_salary_SG']
    MaxSalaryPerPositionPerGames = MaxSalaryPerPositionPerGames.fillna(0)

    AvgSalaryPerPositionPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season','tm', 'game_date', 'Position'])['salary_scale'].max()
        ).reset_index()

    AvgSalaryPerPositionPerGames = AvgSalaryPerPositionPerGames.pivot(index=['id_season', 'tm', 'game_date'], columns='Position', values='salary_scale').reset_index()
    AvgSalaryPerPositionPerGames.columns = ['id_season', 'tm', 'game_date', 'avg_salary_C', 'avg_salary_PF', 'avg_salary_PG', 'avg_salary_SF', 'avg_salary_SG']
    AvgSalaryPerPositionPerGames = AvgSalaryPerPositionPerGames.fillna(0)


    # Averaege expereince per games

    AvgPlayerAttributesPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season','tm', 'game_date'])[['Wt',	'Experience',	'Age',	'cm_size',	'salary_scale']].mean()
        ).reset_index()

    AvgPlayerAttributesPerGames.columns = ['id_season','tm', 'game_date', 'avg_Wt',	'avg_Experience',	'avg_Age',	'avg_cm_size',	'avg_salary']

    # Sum expereince per games
    SumPlayerAttributesPerGames = pd.DataFrame(
        PlayerBoxscores.groupby(['id_season','tm', 'game_date'])[['Wt',	'Experience',	'Age',	'cm_size',	'salary_scale']].sum()
        ).reset_index()

    SumPlayerAttributesPerGames.columns = ['id_season','tm', 'game_date', 'sum_Wt',	'sum_Experience',	'sum_Age',	'sum_cm_size',	'sum_salary']

    PlayerAttributesAtGamesLevel = pd.merge(
        AvgPlayerAttributesPerGames,
        SumPlayerAttributesPerGames,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date']
    )

    PlayerAttributesAtGamesLevel = pd.merge(
        PlayerAttributesAtGamesLevel,
        NumberPlyerPerPositionPerGames,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date']
    )

    PlayerAttributesAtGamesLevel = pd.merge(
        PlayerAttributesAtGamesLevel,
        MaxSalaryPerPositionPerGames,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date']
    )

    PlayerAttributesAtGamesLevel = pd.merge(
        PlayerAttributesAtGamesLevel,
        AvgSalaryPerPositionPerGames,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date']
    )

    #-------------------------------------------------------------------------------------------------
    # Averaege player stats per games shiffted

    PlayerBoxscoresFeatures = PlayerBoxscores[[
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
        PlayerBoxscoresFeatures['before_average_' + col ] = round(PlayerBoxscoresFeatures.groupby(['id_season', 'player_name'])[col].transform(lambda x: x.shift(1).expanding().mean()), 1)

    PlayerBoxscoresFeatures = PlayerBoxscoresFeatures[[
        'id_season', 'tm','player_name', 'game_date',
        'before_average_mp',	'before_average_pts',	'before_average_ast',	'before_average_trb',	'before_average_player_game_score',	'before_average_plus_minus']
        ]

    PlayerBoxscoresFeatures["MinutesRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_mp"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["PtsRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_pts"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["GameScoreRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_player_game_score"].rank("dense", ascending=False)
    PlayerBoxscoresFeatures["PlusMinusRanking"] = PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])["before_average_plus_minus"].rank("dense", ascending=False)

    #------------------------------------------------
    # Averaege player stats per team / games

    AvgPlayerStatsPerGames = pd.DataFrame(
        PlayerBoxscoresFeatures.groupby(['id_season','tm', 'game_date'])[[
            'before_average_player_game_score',	'before_average_plus_minus']].mean()
        ).reset_index()

    SumPlayerStatsPerGames = pd.DataFrame(
        PlayerBoxscoresFeatures.groupby(['id_season', 'tm', 'game_date'])[[
            'before_average_pts',	'before_average_ast',	
            'before_average_trb',	'before_average_player_game_score',	'before_average_plus_minus']].sum()
        ).reset_index()

    SumPlayerStatsPerGames.columns = [
        'id_season', 'tm', 'game_date',
        'before_sum_player_pts',	
        'before_sum_player_ast',	
        'before_sum_player_trb',	
        'before_sum_player_game_score',
        'before_sum_plus_minus']

    PlayerStatsAtGameLevel = pd.merge(
        AvgPlayerStatsPerGames,
        SumPlayerStatsPerGames,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date']
    )

    PlayerStatsAtGameLevel = pd.merge(
        PlayerStatsAtGameLevel,
        PlayerAttributesAtGamesLevel,
        left_on=['id_season',	'tm',	'game_date'],
        right_on=['id_season',	'tm',	'game_date'])

    PlayerStatsAtGameLevel['game_date'] = pd.to_datetime(PlayerStatsAtGameLevel['game_date'])

    return PlayerStatsAtGameLevel


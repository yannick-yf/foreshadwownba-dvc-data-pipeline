
import pandas as pd
import numpy as np
import datetime
import sys
pd.options.mode.chained_assignment = None

def previous_days_average_features(TRAINING_DF):

    #------------------------------
    # STEP 1
    subset_1 = TRAINING_DF
    subset_1['Date'] = pd.to_datetime(subset_1['game_date'])
    #subset_1 = subset_1.set_index('Date')

    #------------------------------
    # STEP 2
    subset_2 = TRAINING_DF[['id_season', 'game_date', 'tm', 'duration_trajet']]

    #------------------------------
    # STEP 3
    subset_1_1 = subset_1.groupby(['id_season', 'tm']).apply(lambda x : x.drop_duplicates('Date').set_index('Date').resample('D').ffill())
    
    # Rename multi index
    subset_1_1.index = subset_1_1.index.set_names(['id_season_index', 'tm_index', 'date'])
    subset_1_1 = subset_1_1.reset_index()

    subset_1_1 = subset_1_1.drop(columns=['id_season_index', 'tm_index'])
    subset_1_1 = subset_1_1[['date', 'id_season', 'game_nb', 'game_date', 'extdom', 'tm', 'opp']]

    LastDaysFeatures =  pd.merge(
        subset_1_1,
        subset_2,
        how='left',
        left_on=['id_season', 'game_date', 'tm'],
        right_on=['id_season', 'game_date', 'tm'])

    #------------------------------
    # STEP 4

    # New method - WORKING
    LastDaysFeatures['game_y_n'] = np.where( LastDaysFeatures['date'] == LastDaysFeatures['game_date'], 1, 0 )

    LastDaysFeatures['ext_y_n'] = np.where( (LastDaysFeatures['date'] == LastDaysFeatures['game_date']) & (LastDaysFeatures['extdom'] =='ext'), 1, 0 )
    LastDaysFeatures['dom_y_n'] = np.where( (LastDaysFeatures['date'] == LastDaysFeatures['game_date']) & (LastDaysFeatures['extdom'] =='dom'), 1, 0 )

    LastDaysFeatures['duration_trajet_y_n'] = np.where( (LastDaysFeatures['date'] == LastDaysFeatures['game_date']) , LastDaysFeatures['duration_trajet'], 0 )

    #---------------------------------------------------------------------------------------

    # LastDaysFeatures['nb_games_last_5days'] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['game_y_n'].transform(lambda x: x.shift(1).expanding().sum()), 1)
    # LastDaysFeatures['nb_games_last_5days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(5)['game_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_games_last_7days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(7)['game_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_games_last_10days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(10)['game_y_n'].sum().droplevel(level=[0,1])

    # LastDaysFeatures['nb_ext_games_last_5days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(5)['ext_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_ext_games_last_7days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(10)['ext_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_ext_games_last_10days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(10)['ext_y_n'].sum().droplevel(level=[0,1])

    # LastDaysFeatures['nb_dom_games_last_5days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(5)['dom_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_dom_games_last_7days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(7)['dom_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['nb_dom_games_last_10days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(10)['dom_y_n'].sum().droplevel(level=[0,1])

    # LastDaysFeatures['sum_duration_trajet_y_n_last_5days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(5)['duration_trajet_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['sum_duration_trajet_y_n_last_7days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(7)['duration_trajet_y_n'].sum().droplevel(level=[0,1])
    # LastDaysFeatures['sum_duration_trajet_y_n_10days'] = LastDaysFeatures.groupby(['id_season', 'tm']).rolling(10)['duration_trajet_y_n'].sum().droplevel(level=[0,1]) 

    LastDaysFeatures['nb_games_last_5days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['game_y_n'].transform(lambda x: x.rolling(5).sum()), 1)
    LastDaysFeatures['nb_games_last_7days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['game_y_n'].transform(lambda x: x.rolling(7).sum()), 1)
    LastDaysFeatures['nb_games_last_10days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['game_y_n'].transform(lambda x: x.rolling(10).sum()), 1)

    LastDaysFeatures['nb_ext_games_last_5days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['ext_y_n'].transform(lambda x: x.rolling(5).sum()), 1)
    LastDaysFeatures['nb_ext_games_last_7days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['ext_y_n'].transform(lambda x: x.rolling(7).sum()), 1)
    LastDaysFeatures['nb_ext_games_last_10days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['ext_y_n'].transform(lambda x: x.rolling(10).sum()), 1)

    LastDaysFeatures['nb_dom_games_last_5days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['dom_y_n'].transform(lambda x: x.rolling(5).sum()), 1)
    LastDaysFeatures['nb_dom_games_last_7days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['dom_y_n'].transform(lambda x: x.rolling(7).sum()), 1)
    LastDaysFeatures['nb_dom_games_last_10days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['dom_y_n'].transform(lambda x: x.rolling(10).sum()), 1)

    LastDaysFeatures['sum_duration_trajet_y_n_last_5days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['duration_trajet_y_n'].transform(lambda x: x.rolling(5).sum()), 1)
    LastDaysFeatures['sum_duration_trajet_y_n_last_7days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['duration_trajet_y_n'].transform(lambda x: x.rolling(7).sum()), 1)
    LastDaysFeatures['sum_duration_trajet_y_n_10days' ] = round(LastDaysFeatures.groupby(['id_season', 'tm'])['duration_trajet_y_n'].transform(lambda x: x.rolling(10).sum()), 1)


    #---------------------------------------------------------------------------------------

    # Only keep the game date
    LastDaysFeatures = LastDaysFeatures.drop('date', axis=1).drop_duplicates(subset=['id_season', 'game_nb', 'game_date', 'extdom', 'tm', 'opp'], keep='first')
    LastDaysFeatures = LastDaysFeatures.drop(['game_nb', 'extdom', 'duration_trajet', 'game_y_n', 'duration_trajet_y_n'], axis = 1)

    # Interpretation: 
    # if nb_games_last_5days=4 that means today is the 4th games in 5 days fro the tm team

    #-----------------------------------
    # Last steps
    TRAINING_DF =  pd.merge(
        TRAINING_DF,
        LastDaysFeatures,
        how='left',
        left_on = ['id_season', 'game_date', 'tm', 'opp'],
        right_on=['id_season', 'game_date', 'tm', 'opp'])

    #-----------------------------------

    TRAINING_DF['sum_duration_trajet_y_n_last_5days'] = TRAINING_DF['sum_duration_trajet_y_n_last_5days' ].fillna(TRAINING_DF['duration_trajet'])

    TRAINING_DF['sum_duration_trajet_y_n_last_7days'] = TRAINING_DF['sum_duration_trajet_y_n_last_7days'].fillna(TRAINING_DF['sum_duration_trajet_y_n_last_5days'])
    TRAINING_DF['sum_duration_trajet_y_n_last_7days'] = TRAINING_DF['sum_duration_trajet_y_n_last_7days'].fillna(TRAINING_DF['duration_trajet'])

    TRAINING_DF['sum_duration_trajet_y_n_10days'] = TRAINING_DF['sum_duration_trajet_y_n_10days'].fillna(TRAINING_DF['sum_duration_trajet_y_n_last_7days'])
    TRAINING_DF['sum_duration_trajet_y_n_10days'] = TRAINING_DF['sum_duration_trajet_y_n_10days'].fillna(TRAINING_DF['sum_duration_trajet_y_n_last_5days'])
    TRAINING_DF['sum_duration_trajet_y_n_10days'] = TRAINING_DF['sum_duration_trajet_y_n_10days'].fillna(TRAINING_DF['duration_trajet'])

    #-------------------------------

    TRAINING_DF['nb_dom_games_last_5days'] = TRAINING_DF['nb_dom_games_last_5days' ].fillna(TRAINING_DF['dom_y_n'])

    TRAINING_DF['nb_dom_games_last_7days'] = TRAINING_DF['nb_dom_games_last_7days'].fillna(TRAINING_DF['nb_dom_games_last_5days'])
    TRAINING_DF['nb_dom_games_last_7days'] = TRAINING_DF['nb_dom_games_last_7days'].fillna(TRAINING_DF['dom_y_n'])

    TRAINING_DF['nb_dom_games_last_10days'] = TRAINING_DF['nb_dom_games_last_10days'].fillna(TRAINING_DF['nb_dom_games_last_7days'])
    TRAINING_DF['nb_dom_games_last_10days'] = TRAINING_DF['nb_dom_games_last_10days'].fillna(TRAINING_DF['nb_dom_games_last_5days'])
    TRAINING_DF['nb_dom_games_last_10days'] = TRAINING_DF['nb_dom_games_last_10days'].fillna(TRAINING_DF['dom_y_n'])

    #-------------------------------

    TRAINING_DF['nb_ext_games_last_5days'] = TRAINING_DF['nb_ext_games_last_5days' ].fillna(TRAINING_DF['ext_y_n'])

    TRAINING_DF['nb_ext_games_last_7days'] = TRAINING_DF['nb_ext_games_last_7days'].fillna(TRAINING_DF['nb_ext_games_last_5days'])
    TRAINING_DF['nb_ext_games_last_7days'] = TRAINING_DF['nb_ext_games_last_7days'].fillna(TRAINING_DF['ext_y_n'])

    TRAINING_DF['nb_ext_games_last_10days'] = TRAINING_DF['nb_ext_games_last_10days'].fillna(TRAINING_DF['nb_ext_games_last_7days'])
    TRAINING_DF['nb_ext_games_last_10days'] = TRAINING_DF['nb_ext_games_last_10days'].fillna(TRAINING_DF['nb_ext_games_last_5days'])
    TRAINING_DF['nb_ext_games_last_10days'] = TRAINING_DF['nb_ext_games_last_10days'].fillna(TRAINING_DF['ext_y_n'])

    #-------------------------------

    TRAINING_DF['nb_games_last_5days'] = TRAINING_DF['nb_games_last_5days' ].fillna(TRAINING_DF['game_nb'])

    TRAINING_DF['nb_games_last_7days'] = TRAINING_DF['nb_games_last_7days'].fillna(TRAINING_DF['nb_games_last_5days'])
    TRAINING_DF['nb_games_last_7days'] = TRAINING_DF['nb_games_last_7days'].fillna(TRAINING_DF['game_nb'])

    TRAINING_DF['nb_games_last_10days'] = TRAINING_DF['nb_games_last_10days'].fillna(TRAINING_DF['nb_games_last_7days'])
    TRAINING_DF['nb_games_last_10days'] = TRAINING_DF['nb_games_last_10days'].fillna(TRAINING_DF['nb_games_last_5days'])
    TRAINING_DF['nb_games_last_10days'] = TRAINING_DF['nb_games_last_10days'].fillna(TRAINING_DF['game_nb'])

    TRAINING_DF = TRAINING_DF.drop(['Date','ext_y_n', 'dom_y_n'], axis = 1)

    return TRAINING_DF


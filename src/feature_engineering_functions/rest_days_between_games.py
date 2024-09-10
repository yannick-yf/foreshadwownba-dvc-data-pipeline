

import pandas as pd 
pd.options.mode.chained_assignment = None

def rest_days_between_games(TRAINING_DF):
    TRAINING_DF['game_date_lag'] = TRAINING_DF.groupby(['id_season', 'tm'])['game_date'].shift(1)
    TRAINING_DF[['game_date_lag','game_date']] = TRAINING_DF[['game_date_lag','game_date']].apply(pd.to_datetime) #if conversion required
    TRAINING_DF['rest'] = (TRAINING_DF['game_date'] - TRAINING_DF['game_date_lag']).dt.days
    TRAINING_DF = TRAINING_DF.drop(columns=['game_date_lag'])
    return TRAINING_DF

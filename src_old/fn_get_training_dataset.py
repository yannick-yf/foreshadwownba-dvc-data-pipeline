# fn_get_nba_games_with_features.py

import pandas as pd

def fn_get_training_dataset():

    #--------------------------------------
    # Pull data from database
    # training_dataset = pd.read_csv('./foreshadwownba-data-engineering-pipeline-output/final_training_dataset_2022-01-20.csv') 
    training_dataset = pd.read_csv('./foreshadwownba-data-engineering-pipeline-output/final_training_dataset_2022-03-30.csv')
    print('Number of rows -  2 lines per game: ' + str(training_dataset.shape[0]))

    return training_dataset
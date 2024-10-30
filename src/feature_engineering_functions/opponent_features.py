"""Module for final data cleaning operations."""

import pandas as pd

def get_opponent_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    """
    #--------------------------------
    # Baseline Classifier - Simple Domain expert rules

    opponent_features = ['id', 'tm' ,'opp', 'before_average_W_ratio']

    training_df_opp = training_df[opponent_features]

    training_df = pd.merge(
        training_df,
        training_df_opp,
        how='left',
        left_on=['id', 'tm' ,'opp'],
        right_on=['id', 'opp', 'tm']
        )
    
    training_df = rename_opponent_columns(training_df)

    return training_df

def rename_opponent_columns(training_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    """
    training_df.columns = training_df.columns.str.replace('_y', '_opp')
    training_df.columns = training_df.columns.str.replace('_x', '')

    return training_df


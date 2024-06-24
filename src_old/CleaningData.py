import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

def CleaningData(TRAINING_DF):
    TRAINING_DF['overtime'] = TRAINING_DF['overtime'].fillna('NOT')
    TRAINING_DF['extdom'] = np.where(TRAINING_DF['extdom']=='@', 'ext', 'dom')

    # Delte games during Covid in the Bubble
    TRAINING_DF = TRAINING_DF.drop(TRAINING_DF[(TRAINING_DF['game_date'] > '2020-03-10') & (TRAINING_DF['id_season'] == 2020)].index)

    return TRAINING_DF


import pandas as pd
import numpy as np
# import datetime
from datetime import datetime

def exctract_days_of_week_from_date(TRAINING_DF):
    
    # The day of the week with Monday=0, Sunday=6.
    # TRAINING_DF['day_of_week'] = pd.to_datetime(TRAINING_DF['game_date']).dt.dayofweek
    
    TRAINING_DF['day_of_week'] = pd.to_datetime(TRAINING_DF['game_date']).dt.day_name()
    return TRAINING_DF

def game_on_weekend_features(TRAINING_DF):
    
    TRAINING_DF['week_weekend'] = np.where(
        TRAINING_DF['day_of_week'].isin(['Saturday', 'Sunday']),
        'weekend',
        'week')

    return TRAINING_DF
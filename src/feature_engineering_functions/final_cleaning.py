import pandas as pd


def final_cleaning(TRAINING_DF):

    TRAINING_DF = TRAINING_DF[TRAINING_DF["id_season"] > TRAINING_DF["id_season"].min()]

    return TRAINING_DF

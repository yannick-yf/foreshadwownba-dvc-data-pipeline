import pandas as pd


def last_game_overtime(TRAINING_DF):

    TRAINING_DF["last_game_overtime"] = TRAINING_DF.groupby(["id_season", "tm"])[
        "overtime"
    ].shift(1)

    return TRAINING_DF

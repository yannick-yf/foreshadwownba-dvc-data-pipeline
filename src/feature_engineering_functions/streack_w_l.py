import pandas as pd
import numpy as np


def streack_w_l(TRAINING_DF):

    TRAINING_DF["streak_w_l_2"] = (
        TRAINING_DF["streak_w_l"].str.extract("(\d+)").astype(int)
    )
    TRAINING_DF["streak_w_l_2"] = np.where(
        TRAINING_DF["streak_w_l"].str.slice(0, 1) == "L",
        TRAINING_DF["streak_w_l_2"] * -1,
        TRAINING_DF["streak_w_l_2"],
    )

    TRAINING_DF["before_streak_w_l"] = TRAINING_DF.groupby(["id_season", "tm"])[
        "streak_w_l_2"
    ].shift(1)

    return TRAINING_DF

# AverageRatioWLExtGame.py

import pandas as pd
import numpy as np


def average_ratio_win_loose_ext_game(TrainingSet):

    TrainingSet["dummy_ext"] = np.where(TrainingSet["extdom"] == "@", 1, 0)
    TrainingSet["Win_at_ext"] = np.where(
        (TrainingSet["dummy_ext"] == 1) & (TrainingSet["results"] == "W"), 1, 0
    )
    TrainingSet["cumsum_ext"] = TrainingSet.groupby(["id_season", "tm"])[
        "dummy_ext"
    ].cumsum()
    TrainingSet["cumsum_W_ext"] = TrainingSet.groupby(["id_season", "tm"])[
        "Win_at_ext"
    ].cumsum()
    TrainingSet["Ratio_Win_Ext"] = (
        TrainingSet["cumsum_W_ext"] / TrainingSet["cumsum_ext"]
    )
    TrainingSet["before_ratio_win_ext"] = TrainingSet["Ratio_Win_Ext"].shift(1)
    TrainingSet["before_ratio_win_ext"] = TrainingSet["before_ratio_win_ext"].fillna(0)

    TrainingSet["dummy_dom"] = np.where(TrainingSet["extdom"] != "@", 1, 0)
    TrainingSet["Win_at_dom"] = np.where(
        (TrainingSet["dummy_dom"] == 1) & (TrainingSet["results"] == "W"), 1, 0
    )
    TrainingSet["cumsum_dom"] = TrainingSet.groupby(["id_season", "tm"])[
        "dummy_dom"
    ].cumsum()
    TrainingSet["cumsum_W_dom"] = TrainingSet.groupby(["id_season", "tm"])[
        "Win_at_dom"
    ].cumsum()
    TrainingSet["Ratio_Win_Dom"] = (
        TrainingSet["cumsum_W_dom"] / TrainingSet["cumsum_dom"]
    )
    TrainingSet["before_ratio_win_dom"] = TrainingSet["Ratio_Win_Dom"].shift(1)
    TrainingSet["before_ratio_win_dom"] = TrainingSet["before_ratio_win_dom"].fillna(0)

    TrainingSet = TrainingSet.drop(
        [
            "dummy_ext",
            "cumsum_W_ext",
            "cumsum_ext",
            "Ratio_Win_Ext",
            "Win_at_ext",
            "Win_at_dom",
            "dummy_dom",
            "cumsum_W_dom",
            "cumsum_dom",
            "Ratio_Win_Dom",
        ],
        axis=1,
    )

    return TrainingSet

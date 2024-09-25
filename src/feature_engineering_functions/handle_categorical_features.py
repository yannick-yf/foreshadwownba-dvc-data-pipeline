"""Module for final data cleaning operations."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def handle_categorical_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """ """
    training_df["last_game_overtime"] = np.where(
        training_df["last_game_overtime"].isin(["2OT", "3OT", "4OT", "5OT"]),
        "Multiple_OT",
        training_df["last_game_overtime"],
    )

    drop_binary_enc = OneHotEncoder(drop="if_binary").fit(
        training_df[["extdom", "week_weekend", "last_game_overtime"]]
    )

    dummies_variables = drop_binary_enc.transform(
        training_df[["extdom", "week_weekend", "last_game_overtime"]]
    ).toarray()

    dummies_variables_df = pd.DataFrame(
        dummies_variables, columns=drop_binary_enc.get_feature_names_out()
    )

    dummies_variables_df = dummies_variables_df.drop("last_game_overtime_NOT", axis=1)

    training_df = pd.concat([training_df, dummies_variables_df], axis=1)

    return training_df

"""Module for final data cleaning operations."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def handle_different_values_overtime(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process to handle the different values of overtime columns
    """
    training_df["last_game_overtime"] = np.where(
        training_df["last_game_overtime"].isin(["2OT", "3OT", "4OT", "5OT"]),
        "Multiple_OT",
        training_df["last_game_overtime"],
    )

    return training_df


def handle_categorical_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline to handle categorical features
    """
    training_df = handle_different_values_overtime(training_df)

    drop_binary_enc = OneHotEncoder(drop="if_binary").fit(
        training_df[["extdom", "last_game_overtime"]]
    )

    dummies_variables = drop_binary_enc.transform(
        training_df[["extdom", "last_game_overtime"]]
    ).toarray()

    dummies_variables_df = pd.DataFrame(
        dummies_variables, columns=drop_binary_enc.get_feature_names_out(), dtype=int
    )

    training_df = pd.concat([training_df, dummies_variables_df], axis=1)

    training_df.drop(["extdom", "last_game_overtime"], axis=1)

    return training_df

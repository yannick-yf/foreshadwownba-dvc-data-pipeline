"""Module for calculating win-loss streak features."""

import numpy as np
import pandas as pd


def calculate_streak_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win-loss streak features for each team in each season.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data with 'streak_w_l' column.

    Returns:
        pd.DataFrame: DataFrame with new streak-related columns.
    """
    training_df = training_df.copy()

    # Extract the numeric value from the streak_w_l column
    training_df["streak_w_l_2"] = (
        training_df["streak_w_l"].str.extract(r"(\d+)").astype(int)
    )

    # Adjust the streak value to be negative for losses
    training_df["streak_w_l_2"] = np.where(
        training_df["streak_w_l"].str[0] == "L",
        training_df["streak_w_l_2"] * -1,
        training_df["streak_w_l_2"],
    )

    # Calculate the streak value from the previous game
    training_df["before_streak_w_l"] = training_df.groupby(["id_season", "tm"])[
        "streak_w_l_2"
    ].shift(1)

    # Drop the non needed features
    training_df = training_df.drop(["streak_w_l_2"], axis=1)

    return training_df

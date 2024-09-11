"""Module for calculating win-loss streak features."""

import pandas as pd
import numpy as np

def calculate_streak_features(training_df):
    """
    Calculate win-loss streak features for each team in each season.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data with 'streak_w_l' column.

    Returns:
        pd.DataFrame: DataFrame with new streak-related columns.
    """
    df = training_df.copy()

    # Extract the numeric value from the streak_w_l column
    df["streak_w_l_2"] = df["streak_w_l"].str.extract(r"(\d+)").astype(int)

    # Adjust the streak value to be negative for losses
    df["streak_w_l_2"] = np.where(
        df["streak_w_l"].str[0] == "L",
        df["streak_w_l_2"] * -1,
        df["streak_w_l_2"]
    )

    # Calculate the streak value from the previous game
    df["before_streak_w_l"] = df.groupby(["id_season", "tm"])["streak_w_l_2"].shift(1)

    return df

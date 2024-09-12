"""Module for date-related feature engineering functions."""

import pandas as pd
import numpy as np


def extract_days_of_week_from_date(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the day of the week from the game date.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with a new 'day_of_week' column.
    """
    training_df["day_of_week"] = pd.to_datetime(training_df["game_date"]).dt.day_name()
    return training_df


def game_on_weekend_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature indicating whether the game is on a weekend or weekday.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data with 'day_of_week' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'week_weekend' column.
    """
    training_df["week_weekend"] = np.where(
        training_df["day_of_week"].isin(["Saturday", "Sunday"]), "weekend", "week"
    )
    return training_df

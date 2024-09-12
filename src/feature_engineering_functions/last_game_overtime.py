"""Module for calculating features related to overtime in previous games."""

import pandas as pd


def last_game_overtime(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate whether the last game for each team in each season went to overtime.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with a new 'last_game_overtime' column.
    """
    training_df["last_game_overtime"] = training_df.groupby(["id_season", "tm"])[
        "overtime"
    ].shift(1)

    return training_df

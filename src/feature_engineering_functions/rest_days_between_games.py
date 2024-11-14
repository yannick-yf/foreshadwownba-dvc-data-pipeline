"""Module for calculating rest days between games."""

import pandas as pd

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def calculate_rest_days_between_games(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of rest days between games for each team in each season.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with a new 'rest' column indicating days between games.
    """
    # Create a copy to avoid modifying the original DataFrame
    training_df = training_df.copy()

    # Calculate the previous game date for each team in each season
    training_df["game_date_lag"] = training_df.groupby(["id_season", "tm"])[
        "game_date"
    ].shift(1)

    # Convert game dates to datetime if not already
    training_df[["game_date_lag", "game_date"]] = training_df[
        ["game_date_lag", "game_date"]
    ].apply(pd.to_datetime)

    # Calculate the number of days between games
    training_df["rest"] = (
        training_df["game_date"] - training_df["game_date_lag"]
    ).dt.days

    # When game_nb = 1 , rest value is na
    training_df["rest"] = training_df["rest"].fillna(0)

    training_df["rest"] = training_df["rest"].astype(int)

    # Remove the temporary column
    training_df = training_df.drop(columns=["game_date_lag"])

    return training_df

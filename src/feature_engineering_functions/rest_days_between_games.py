"""Module for calculating rest days between games."""

import pandas as pd

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def calculate_rest_days_between_games(training_df):
    """
    Calculate the number of rest days between games for each team in each season.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with a new 'rest' column indicating days between games.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = training_df.copy()

    # Calculate the previous game date for each team in each season
    df["game_date_lag"] = df.groupby(["id_season", "tm"])["game_date"].shift(1)

    # Convert game dates to datetime if not already
    df[["game_date_lag", "game_date"]] = df[["game_date_lag", "game_date"]].apply(pd.to_datetime)

    # Calculate the number of days between games
    df["rest"] = (df["game_date"] - df["game_date_lag"]).dt.days

    # Remove the temporary column
    df = df.drop(columns=["game_date_lag"])

    return df

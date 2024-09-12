"""Module for final data cleaning operations."""

import pandas as pd


def final_cleaning(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform final cleaning on the training DataFrame.

    This function removes the earliest season from the dataset.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing training data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with the earliest season removed.
    """
    return training_df[training_df["id_season"] > training_df["id_season"].min()]

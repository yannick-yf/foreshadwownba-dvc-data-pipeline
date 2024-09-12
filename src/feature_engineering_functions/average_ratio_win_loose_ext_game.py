"""
Calculate average ratio of wins for external and domestic games.
"""

import numpy as np
import pandas as pd


def average_ratio_win_loose_ext_game(training_set: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average ratio of wins for external and domestic games.

    Args:
        training_set (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: Modified DataFrame with new calculated columns.
    """
    # External game calculations
    training_set["dummy_ext"] = np.where(training_set["extdom"] == "ext", 1, 0)
    training_set["win_at_ext"] = np.where(
        (training_set["dummy_ext"] == 1) & (training_set["results"] == "W"), 1, 0
    )
    training_set["cumsum_ext"] = training_set.groupby(["id_season", "tm"])[
        "dummy_ext"
    ].cumsum()
    training_set["cumsum_w_ext"] = training_set.groupby(["id_season", "tm"])[
        "win_at_ext"
    ].cumsum()
    training_set["ratio_win_ext"] = (
        training_set["cumsum_w_ext"] / training_set["cumsum_ext"]
    )
    training_set["before_ratio_win_ext"] = (
        training_set["ratio_win_ext"].shift(1).fillna(0)
    )

    # Domestic game calculations
    training_set["dummy_dom"] = np.where(training_set["extdom"] != "ext", 1, 0)
    training_set["win_at_dom"] = np.where(
        (training_set["dummy_dom"] == 1) & (training_set["results"] == "W"), 1, 0
    )
    training_set["cumsum_dom"] = training_set.groupby(["id_season", "tm"])[
        "dummy_dom"
    ].cumsum()
    training_set["cumsum_w_dom"] = training_set.groupby(["id_season", "tm"])[
        "win_at_dom"
    ].cumsum()
    training_set["ratio_win_dom"] = (
        training_set["cumsum_w_dom"] / training_set["cumsum_dom"]
    )
    training_set["before_ratio_win_dom"] = (
        training_set["ratio_win_dom"].shift(1).fillna(0)
    )

    # Drop intermediate columns
    columns_to_drop = [
        "dummy_ext",
        "cumsum_w_ext",
        "cumsum_ext",
        "ratio_win_ext",
        "win_at_ext",
        "win_at_dom",
        "dummy_dom",
        "cumsum_w_dom",
        "cumsum_dom",
        "ratio_win_dom",
    ]
    training_set = training_set.drop(columns=columns_to_drop)

    return training_set

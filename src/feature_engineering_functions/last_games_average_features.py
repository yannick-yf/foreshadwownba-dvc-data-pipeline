"""Module for calculating average features from previous games."""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def previous_games_average_features(
        training_df: pd.DataFrame,
        columns_to_process = ["pts_tm", "pts_opp"]
        ) -> pd.DataFrame:
    """
    Calculate average features from previous games.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with new average features.
    """
    training_df = training_df.sort_values(by=["id_season", "tm", "game_nb"])

    # columns_to_process = ["pts_tm", "pts_opp"]

    for col in columns_to_process:
        training_df[f"before_average_{col}"] = round(
            training_df.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).expanding().mean()
            ),
            1,
        )
        training_df[f"before_average_lastfivegame_{col}"] = round(
            training_df.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(5).mean()
            ),
            1,
        )
        training_df[f"before_average_lasttengame_{col}"] = round(
            training_df.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(10).mean()
            ),
            1,
        )

        # Fill na values
        training_df[f"before_average_lastfivegame_{col}"] = training_df[
            f"before_average_lastfivegame_{col}"
        ].fillna(training_df[f"before_average_{col}"])
        training_df[f"before_average_lasttengame_{col}"] = training_df[
            f"before_average_lasttengame_{col}"
        ].fillna(training_df[f"before_average_lastfivegame_{col}"])
        training_df[f"before_average_lasttengame_{col}"] = training_df[
            f"before_average_lasttengame_{col}"
        ].fillna(training_df[f"before_average_{col}"])

        training_df[f"diff_all_minus_lastfivegame_{col}"] = (
            training_df[f"before_average_{col}"]
            - training_df[f"before_average_lastfivegame_{col}"]
        )
        training_df[f"diff_all_minus_lasttengame_{col}"] = (
            training_df[f"before_average_{col}"]
            - training_df[f"before_average_lasttengame_{col}"]
        )

    return training_df


def previous_games_ratio_average_features(training_df):
    """
    Calculate ratio average features from previous games.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with new ratio average features.
    """
    
    training_df_subset = training_df[
        ["id_season", "game_nb", "game_date", "extdom", "tm", "opp", "results"]
    ]
    training_df_subset["w_for_ratio"] = np.where(
        training_df_subset["results"] == "W", 1, 0
    )
    training_df_subset["cum_sum_win"] = training_df_subset.groupby(["id_season", "tm"])[
        "w_for_ratio"
    ].cumsum()
    training_df_subset["average_ratio"] = (
        training_df_subset["cum_sum_win"] / training_df_subset["game_nb"]
    )

    training_df_subset["before_average_W_ratio"] = training_df_subset.groupby(
        ["id_season", "tm"]
    )["average_ratio"].shift(1)

    df_ratio_last_5_games = calculate_rolling_ratio(training_df_subset, window=5)
    df_ratio_last_10_games = calculate_rolling_ratio(training_df_subset, window=10)

    df_ratio_last = pd.merge(
        df_ratio_last_5_games,
        df_ratio_last_10_games,
        how="left",
        on=["id_season", "game_date", "tm"],
    )

    final = pd.merge(
        training_df_subset,
        df_ratio_last,
        how="left",
        on=["id_season", "game_date", "tm"],
    )

    fill_na_values(final)
    calculate_diff_ratios(final)

    columns_to_drop = [
        "game_nb",
        "extdom",
        "results",
        "w_for_ratio",
        "cum_sum_win",
        "average_ratio",
        "w_for_ratio_x",
        "w_for_ratio_y",
    ]
    final = final.drop(columns=columns_to_drop)

    return pd.merge(
        training_df,
        final,
        how="left",
        on=["id_season", "game_date", "tm", "opp"],
    )


def calculate_rolling_ratio(training_df, window):
    """
    Calculate rolling ratio for a given window size.

    Args:
        df (pd.DataFrame): Input DataFrame.
        window (int): Size of the rolling window.

    Returns:
        pd.DataFrame: DataFrame with calculated rolling ratio.
    """
    suffix = "five" if window == 5 else "ten"
    training_df_ratio = (
        training_df.set_index("game_date")
        .groupby(["id_season", "tm"])
        .rolling(window)["w_for_ratio"]
        .sum()
        .reset_index()
    )
    training_df_ratio[f"before_average_last{suffix}game_W_ratio"] = (
        training_df_ratio["w_for_ratio"] / window
    )
    training_df_ratio[
        f"before_average_last{suffix}game_W_ratio"
    ] = training_df_ratio.groupby(["id_season", "tm"])[
        f"before_average_last{suffix}game_W_ratio"
    ].shift(
        1
    )
    return training_df_ratio


def fill_na_values(training_df):
    """
    Fill NA values in the DataFrame.

    Args:
        training_df (pd.DataFrame): Input DataFrame.
    """
    training_df["before_average_lastfivegame_W_ratio"] = training_df[
        "before_average_lastfivegame_W_ratio"
    ].fillna(training_df["before_average_W_ratio"])
    training_df["before_average_lasttengame_W_ratio"] = training_df[
        "before_average_lasttengame_W_ratio"
    ].fillna(training_df["before_average_lastfivegame_W_ratio"])
    training_df["before_average_lasttengame_W_ratio"] = training_df[
        "before_average_lasttengame_W_ratio"
    ].fillna(training_df["before_average_W_ratio"])


def calculate_diff_ratios(training_df):
    """
    Calculate difference ratios.

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    training_df["diff_all_minus_lastfivegame_W_ratio"] = (
        training_df["before_average_W_ratio"]
        - training_df["before_average_lastfivegame_W_ratio"]
    )
    training_df["diff_all_minus_lasttengame_W_ratio"] = (
        training_df["before_average_W_ratio"]
        - training_df["before_average_lasttengame_W_ratio"]
    )


def previous_season_ratio_features(training_df):
    """
    Calculate ratio features based on the previous season's performance.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with new previous season ratio features.
    """
    max_game_id = calculate_max_game_id(training_df)
    training_df = pd.merge(training_df, max_game_id, how="left", on=["id_season", "tm"])

    last_season_ratio_features = calculate_last_season_ratio(training_df)

    training_df = pd.merge(
        training_df,
        last_season_ratio_features,
        how="left",
        left_on=["id_season", "tm"],
        right_on=["id_season_plus1", "tm"],
    )

    training_df = training_df.drop(columns=["id_season_plus1", "max_game_id"])
    training_df["before_season_ratio"] = training_df["before_season_ratio"].fillna(0)

    return training_df


def calculate_max_game_id(training_df):
    """
    Calculate the maximum game ID for each season and team.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with maximum game IDs.
    """
    max_game_id = (
        training_df.groupby(["id_season", "tm"])["game_nb"].max().reset_index()
    )
    max_game_id.columns = ["id_season", "tm", "max_game_id"]
    return max_game_id


def calculate_last_season_ratio(training_df):
    """
    Calculate the ratio features for the last game of each season.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with last season ratio features.
    """
    last_season_ratio = training_df[
        training_df["game_nb"] == training_df["max_game_id"]
    ]
    last_season_ratio["before_season_ratio"] = (
        last_season_ratio["w_tot"] / last_season_ratio["max_game_id"]
    )
    last_season_ratio["id_season_plus1"] = last_season_ratio["id_season"] + 1
    return last_season_ratio[["id_season_plus1", "tm", "before_season_ratio"]]

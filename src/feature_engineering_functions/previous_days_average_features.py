"""
Previous days features computation
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def previous_days_average_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate games average features from previous days.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: DataFrame with new ratio average features.
    """

    # ------------------------------
    # STEP 1
    subset_1 = training_df
    subset_1["Date"] = pd.to_datetime(subset_1["game_date"])

    # ------------------------------
    # STEP 2
    subset_2 = training_df[["id_season", "game_date", "tm", "duration_trip"]]

    # ------------------------------
    # STEP 3
    subset_1_1 = subset_1.groupby(["id_season", "tm"]).apply(
        lambda x: x.drop_duplicates("Date").set_index("Date").resample("D").ffill()
    )

    # Rename multi index
    subset_1_1.index = subset_1_1.index.set_names(
        ["id_season_index", "tm_index", "date"]
    )
    subset_1_1 = subset_1_1.reset_index()

    subset_1_1 = subset_1_1.drop(columns=["id_season_index", "tm_index"])
    subset_1_1 = subset_1_1[
        ["date", "id_season", "game_nb", "game_date", "extdom", "tm", "opp"]
    ]

    last_days_features = pd.merge(
        subset_1_1,
        subset_2,
        how="left",
        left_on=["id_season", "game_date", "tm"],
        right_on=["id_season", "game_date", "tm"],
    )

    # ------------------------------
    # STEP 4

    last_days_features = _game_indicator_columns(last_days_features)

    # ------------------------------------------------

    last_days_features = _recode_game_day_trip_duration(last_days_features)

    # ------------------------------------------------
    # Usage example:
    windows_to_calculate = [5, 7, 10]
    last_days_features = _generate_rolling_game_count_features(
        df=last_days_features,
        group_cols=["id_season", "tm"],
        game_indicator_col="game_y_n",
        windows=windows_to_calculate,
    )

    last_days_features = _generate_rolling_away_game_count_features(
        df=last_days_features,
        group_cols=["id_season", "tm"],
        away_game_indicator_col="ext_y_n",
        windows=windows_to_calculate,
    )

    last_days_features = _generate_rolling_home_game_count_features(
        df=last_days_features,
        group_cols=["id_season", "tm"],
        away_game_indicator_col="dom_y_n",
        windows=windows_to_calculate,
    )

    last_days_features = _generate_rolling_trip_duration_sum_features(
        df=last_days_features,
        group_cols=["id_season", "tm"],
        trip_duration_col="duration_trip_y_n",
        windows=windows_to_calculate,
    )

    # ---------------------------------------------------------------------------------------

    # Only keep the game date
    last_days_features = last_days_features.drop("date", axis=1).drop_duplicates(
        subset=["id_season", "game_nb", "game_date", "extdom", "tm", "opp"],
        keep="first",
    )
    last_days_features = last_days_features.drop(
        ["game_nb", "extdom", "duration_trip", "game_y_n", "duration_trip_y_n"],
        axis=1,
    )

    # Interpretation:
    # if nb_games_last_5days=4 that means today is the 4th games in 5 days fro the tm team

    # -----------------------------------
    # Last steps
    training_df = pd.merge(
        training_df,
        last_days_features,
        how="left",
        on=["id_season", "game_date", "tm", "opp"],
    )

    training_df = _fillna_previous_days_average_features(training_df)

    training_df = training_df.drop(["Date", "ext_y_n", "dom_y_n"], axis=1)

    return training_df


def _fillna_previous_days_average_features(training_df: pd.DataFrame) -> pd.DataFrame:
    feature_groups = [
        {
            "prefix": "sum_duration_trip_y_n",
            "days": ["last_5days", "last_7days", "last_10days"],
            "fallback": "duration_trip",
        },
        {
            "prefix": "nb_dom_games",
            "days": ["last_5days", "last_7days", "last_10days"],
            "fallback": "dom_y_n",
        },
        {
            "prefix": "nb_ext_games",
            "days": ["last_5days", "last_7days", "last_10days"],
            "fallback": "ext_y_n",
        },
        {
            "prefix": "nb_games",
            "days": ["last_5days", "last_7days", "last_10days"],
            "fallback": "game_nb",
        },
    ]

    for group in feature_groups:
        prefix = group["prefix"]
        days = group["days"]
        fallback = group["fallback"]

        for i, day in enumerate(days):
            column = f"{prefix}_{day}"
            training_df[column] = training_df[column].fillna(training_df[fallback])

            if i > 0:
                for prev_day in days[:i]:
                    prev_column = f"{prefix}_{prev_day}"
                    training_df[column] = training_df[column].fillna(
                        training_df[prev_column]
                    )

    return training_df


def _game_indicator_columns(last_days_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create indicator columns for games, away games, and home games.

    This function adds three new columns to the DataFrame:
    - 'game_y_n': Indicates whether a game was played on a given date (1 for yes, 0 for no)
    - 'ext_y_n': Indicates whether an away game was played on a given date (1 for yes, 0 for no)
    - 'dom_y_n': Indicates whether a home game was played on a given date (1 for yes, 0 for no)

    Args:
        last_days_features (pd.DataFrame): Input DataFrame containing game data

    Returns:
        pd.DataFrame: DataFrame with new indicator columns added
    """
    # Indicate if a game was played on a given date
    last_days_features["game_y_n"] = np.where(
        last_days_features["date"] == last_days_features["game_date"], 1, 0
    )

    # Indicate if an away game was played on a given date
    last_days_features["ext_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"])
        & (last_days_features["extdom"] == "ext"),
        1,
        0,
    )

    # Indicate if a home game was played on a given date
    last_days_features["dom_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"])
        & (last_days_features["extdom"] == "dom"),
        1,
        0,
    )

    return last_days_features


def _recode_game_day_trip_duration(last_days_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create a column that shows the trip duration on game days and 0 on non-game days.

    This function adds a new column 'duration_trip_y_n' to the DataFrame:
    - On game days (where date matches game_date), it shows the duration of the trip.
    - On non-game days, it shows 0.

    Args:
        last_days_features (pd.DataFrame): Input DataFrame containing game and trip data

    Returns:
        pd.DataFrame: DataFrame with the new 'duration_trip_y_n' column added
    """
    last_days_features["duration_trip_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"]),
        last_days_features["duration_trip"],
        0,
    )

    return last_days_features


def _generate_rolling_game_count_features(
    df: pd.DataFrame, group_cols: list, game_indicator_col: str, windows: list
) -> pd.DataFrame:
    """
    Generate features counting the number of games for the last x days.

    Args:
        df (pd.DataFrame): Input DataFrame containing game data
        group_cols (list): Columns to group by (e.g., ["id_season", "tm"])
        game_indicator_col (str): Column name indicating whether a game was played (e.g., "game_y_n")
        windows (list): List of day windows to calculate (e.g., [5, 7, 10])

    Returns:
        pd.DataFrame: DataFrame with new rolling game count features added
    """
    for window in windows:
        feature_name = f"nb_games_last_{window}days"
        df[feature_name] = (
            df.groupby(group_cols)[game_indicator_col]
            .transform(lambda x: x.rolling(window).sum())
            .round(1)
        )

    return df


def _generate_rolling_away_game_count_features(
    df: pd.DataFrame, group_cols: list, away_game_indicator_col: str, windows: list
) -> pd.DataFrame:
    """
    Generate features counting the number of away games for the last x days.

    Args:
        df (pd.DataFrame): Input DataFrame containing game data
        group_cols (list): Columns to group by (e.g., ["id_season", "tm"])
        away_game_indicator_col (str): Column name indicating whether an away game was played (e.g., "ext_y_n")
        windows (list): List of day windows to calculate (e.g., [5, 7, 10])

    Returns:
        pd.DataFrame: DataFrame with new rolling away game count features added
    """
    for window in windows:
        feature_name = f"nb_ext_games_last_{window}days"
        df[feature_name] = (
            df.groupby(group_cols)[away_game_indicator_col]
            .transform(lambda x: x.rolling(window).sum())
            .round(1)
        )

    return df


def _generate_rolling_home_game_count_features(
    df: pd.DataFrame, group_cols: list, away_game_indicator_col: str, windows: list
) -> pd.DataFrame:
    """
    Generate features counting the number of away games for the last x days.

    Args:
        df (pd.DataFrame): Input DataFrame containing game data
        group_cols (list): Columns to group by (e.g., ["id_season", "tm"])
        home_game_indicator_col (str): Column name indicating whether an home game was played (e.g., "dom_y_n")
        windows (list): List of day windows to calculate (e.g., [5, 7, 10])

    Returns:
        pd.DataFrame: DataFrame with new rolling away game count features added
    """
    for window in windows:
        feature_name = f"nb_dom_games_last_{window}days"
        df[feature_name] = (
            df.groupby(group_cols)[away_game_indicator_col]
            .transform(lambda x: x.rolling(window).sum())
            .round(1)
        )

    return df


def _generate_rolling_trip_duration_sum_features(
    df: pd.DataFrame, group_cols: list, trip_duration_col: str, windows: list
) -> pd.DataFrame:
    """
    Generate features summing the duration of trips for the last x days.

    Args:
        df (pd.DataFrame): Input DataFrame containing trip duration data
        group_cols (list): Columns to group by (e.g., ["id_season", "tm"])
        trip_duration_col (str): Column name indicating the duration of trips (e.g., "duration_trip_y_n")
        windows (list): List of day windows to calculate (e.g., [5, 7, 10])

    Returns:
        pd.DataFrame: DataFrame with new rolling trip duration sum features added
    """
    for window in windows:
        feature_name = f"sum_duration_trip_y_n_last_{window}days"
        df[feature_name] = (
            df.groupby(group_cols)[trip_duration_col]
            .transform(lambda x: x.rolling(window).sum())
            .round(1)
        )

    return df

"""
Duration trip between two cities features
"""

import pandas as pd
import numpy as np


def duration_trip_hours_between_cities(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate duration trip features between cities for NBA games.

    Args:
        training_df (pd.DataFrame): Input DataFrame containing game data.

    Returns:
        pd.DataFrame: Modified DataFrame with new calculated duration features.
    """
    city_name = pd.read_csv("./data/constants/team_name.csv")
    data_city_distance = pd.read_csv("./data/constants/data_city_distance.csv")

    dist_features_df = training_df[
        ["id_season", "game_date", "game_nb", "tm", "opp", "extdom"]
    ]

    dist_features_df = _get_distance_between_cities(
        city_name, data_city_distance, dist_features_df
    )

    dist_features_df = _get_duration_trip_features(dist_features_df)

    dist_features_df = _fillna_trip_features(dist_features_df)

    # Merge with original DataFrame
    training_df = pd.merge(
        training_df,
        dist_features_df,
        how="left",
        on=["id_season", "game_date", "tm"],
    )

    return training_df


def _fillna_trip_features(
    dist_features_df: pd.DataFrame,
) -> pd.DataFrame:
    # Fill na values
    fill_columns = [
        "before_average_lastfivegame_duration_trip",
        "before_average_lasttengame_duration_trip",
        "before_sum_lastfivegame_duration_trip",
        "before_sum_lasttengame_duration_trip",
    ]

    for col in fill_columns:
        dist_features_df[col] = dist_features_df[col].fillna(
            dist_features_df["duration_trip"]
        )

    dist_features_df["before_average_lasttengame_duration_trip"] = dist_features_df[
        "before_average_lasttengame_duration_trip"
    ].fillna(dist_features_df["before_average_lastfivegame_duration_trip"])

    dist_features_df["before_sum_lasttengame_duration_trip"] = dist_features_df[
        "before_sum_lasttengame_duration_trip"
    ].fillna(dist_features_df["before_sum_lastfivegame_duration_trip"])

    return dist_features_df


def _get_duration_trip_features(
    dist_features_df: pd.DataFrame,
) -> pd.DataFrame:

    # Features Creation from duration traject
    group_columns = ["id_season", "tm"]

    dist_features_df["before_average_duration_trip"] = round(
        dist_features_df.groupby(group_columns)["duration_trip"].transform(
            lambda x: x.shift(1).expanding().mean()
        ),
        1,
    )
    dist_features_df["before_average_lastfivegame_duration_trip"] = round(
        dist_features_df.groupby(group_columns)["duration_trip"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        ),
        1,
    )
    dist_features_df["before_average_lasttengame_duration_trip"] = round(
        dist_features_df.groupby(group_columns)["duration_trip"].transform(
            lambda x: x.shift(1).rolling(10).mean()
        ),
        1,
    )
    dist_features_df["before_sum_lastfivegame_duration_trip"] = round(
        dist_features_df.groupby(group_columns)["duration_trip"].transform(
            lambda x: x.shift(1).rolling(5).sum()
        ),
        1,
    )
    dist_features_df["before_sum_lasttengame_duration_trip"] = round(
        dist_features_df.groupby(group_columns)["duration_trip"].transform(
            lambda x: x.shift(1).rolling(10).sum()
        ),
        1,
    )

    return dist_features_df


def _get_distance_between_cities(
    city_name: pd.DataFrame,
    data_city_distance: pd.DataFrame,
    dist_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the distance between two cities based on the provided data.

    Args:
        city_from (str): The city from which the distance is calculated.
        city_to (str): The city to which the distance is calculated.
        data_city_distance (pd.DataFrame): DataFrame containing city distance information.

    Returns:
        float: The distance between the two cities.
    """

    # The idea of this code is to create city_1 and city_2
    # city_1 is the city where the team team come from. 
    # If previous game is ext the city is the opp else this is tm
    # city_2 is the city where the game is.

    dist_features_df["team_abrev_from"] = np.where(
        dist_features_df["game_nb"] == 1,
        dist_features_df["tm"],
        np.where(
            dist_features_df["extdom"].shift(1) == "ext",
            dist_features_df["opp"].shift(1),
            dist_features_df["tm"],
        ),
    )

    dist_features_df["team_abrev_to"] = np.where(
        dist_features_df["extdom"] == "ext",
        dist_features_df["opp"],
        dist_features_df["tm"],
    )

    dist_features_df = pd.merge(
        dist_features_df,
        city_name[["team", "city"]],
        how="left",
        left_on=["team_abrev_from"],
        right_on=["team"],
    )

    dist_features_df = dist_features_df.rename({"city": "city_from"}, axis="columns")

    dist_features_df = dist_features_df.drop(["team"], axis=1)

    dist_features_df = pd.merge(
        dist_features_df,
        city_name[["team", "city"]],
        how="left",
        left_on=["team_abrev_to"],
        right_on=["team"],
    )

    dist_features_df = dist_features_df.rename({"city": "city_to"}, axis="columns")

    dist_features_df = dist_features_df.drop(["team"], axis=1)

    # Add the distance between the two cities

    dist_features_df = pd.merge(
        dist_features_df,
        data_city_distance,
        how="left",
        left_on=["city_from", "city_to"],
        right_on=["city1", "city2"],
    )

    dist_features_df = dist_features_df[
        ["id_season", "game_date", "tm", "duration_trip"]
    ]

    return dist_features_df

# --------------------------------------------
# Duration trip between two cities features

import pandas as pd
import numpy as np


def duration_trip_hours_between_cities(training_df):
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

    dist_features_df["city_1"] = np.where(
        dist_features_df["game_nb"] == 1,
        dist_features_df["tm"],
        np.where(
            dist_features_df["extdom"].shift(1) == "ext",
            dist_features_df["opp"].shift(1),
            dist_features_df["tm"],
        ),
    )

    dist_features_df["city_2"] = np.where(
        dist_features_df["extdom"] == "ext",
        dist_features_df["opp"],
        dist_features_df["tm"],
    )

    dist_features_df = pd.merge(
        dist_features_df,
        city_name[["team", "city"]],
        how="left",
        left_on=["city_1"],
        right_on=["team"],
    )

    dist_features_df = pd.merge(
        dist_features_df,
        city_name[["team", "city"]],
        how="left",
        left_on=["city_2"],
        right_on=["team"],
    )

    dist_features_df = pd.merge(
        dist_features_df,
        data_city_distance,
        how="left",
        left_on=["city_x", "city_y"],
        right_on=["city1", "city2"],
    )

    dist_features_df = dist_features_df[
        ["id_season", "game_date", "tm", "duration_trajet"]
    ]

    # Features Creation from duration traject
    group_columns = ["id_season", "tm"]
    dist_features_df["before_average_duration_trajet"] = round(
        dist_features_df.groupby(group_columns)["duration_trajet"].transform(
            lambda x: x.shift(1).expanding().mean()
        ),
        1,
    )
    dist_features_df["before_average_lastfivegame_duration_trajet"] = round(
        dist_features_df.groupby(group_columns)["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        ),
        1,
    )
    dist_features_df["before_average_lasttengame_duration_trajet"] = round(
        dist_features_df.groupby(group_columns)["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(10).mean()
        ),
        1,
    )
    dist_features_df["before_sum_lastfivegame_duration_trajet"] = round(
        dist_features_df.groupby(group_columns)["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(5).sum()
        ),
        1,
    )
    dist_features_df["before_sum_lasttengame_duration_trajet"] = round(
        dist_features_df.groupby(group_columns)["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(10).sum()
        ),
        1,
    )

    # Fill na values
    fill_columns = [
        "before_average_lastfivegame_duration_trajet",
        "before_average_lasttengame_duration_trajet",
        "before_sum_lastfivegame_duration_trajet",
        "before_sum_lasttengame_duration_trajet",
    ]

    for col in fill_columns:
        dist_features_df[col] = dist_features_df[col].fillna(
            dist_features_df["duration_trajet"]
        )

    dist_features_df["before_average_lasttengame_duration_trajet"] = dist_features_df[
        "before_average_lasttengame_duration_trajet"
    ].fillna(dist_features_df["before_average_lastfivegame_duration_trajet"])

    dist_features_df["before_sum_lasttengame_duration_trajet"] = dist_features_df[
        "before_sum_lasttengame_duration_trajet"
    ].fillna(dist_features_df["before_sum_lastfivegame_duration_trajet"])

    # Merge with original DataFrame
    training_df = pd.merge(
        training_df,
        dist_features_df,
        how="left",
        on=["id_season", "game_date", "tm"],
    )

    return training_df

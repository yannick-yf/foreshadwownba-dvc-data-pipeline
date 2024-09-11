import pandas as pd
import numpy as np

# --------------------------------------------
# Duration trip between two cities features


def duration_trip_hours_between_cities(TRAINING_DF):

    # -----------------------------------
    city_name = pd.read_csv("./data/constants/team_name.csv")
    data_city_distance = pd.read_csv("./data/constants/data_city_distance.csv")

    # -----------------------------------
    DistFeaturesDF = TRAINING_DF[
        ["id_season", "game_date", "game_nb", "tm", "opp", "extdom"]
    ]

    # -----------------------------------
    DistFeaturesDF["city_1"] = np.where(
        DistFeaturesDF["game_nb"] == 1,
        DistFeaturesDF["tm"],
        np.where(
            DistFeaturesDF["extdom"].shift(1) == "ext",
            DistFeaturesDF["opp"].shift(1),
            DistFeaturesDF["tm"],
        ),
    )

    DistFeaturesDF["city_2"] = np.where(
        DistFeaturesDF["extdom"] == "ext", DistFeaturesDF["opp"], DistFeaturesDF["tm"]
    )

    # -----------------------------------
    DistFeaturesDF = pd.merge(
        DistFeaturesDF,
        city_name[["team", "city"]],
        how="left",
        left_on=["city_1"],
        right_on=["team"],
    )

    DistFeaturesDF = pd.merge(
        DistFeaturesDF,
        city_name[["team", "city"]],
        how="left",
        left_on=["city_2"],
        right_on=["team"],
    )

    # -----------------------------------
    DistFeaturesDF = pd.merge(
        DistFeaturesDF,
        data_city_distance,
        how="left",
        left_on=["city_x", "city_y"],
        right_on=["city1", "city2"],
    )

    DistFeaturesDF = DistFeaturesDF[["id_season", "game_date", "tm", "duration_trajet"]]

    # -----------------------------------
    # Features Creation from duration traject

    # DistFeaturesDF['before_average_duration_trajet'] = round(DistFeaturesDF.groupby(['id_season', 'tm']).expanding()['duration_trajet'].mean().droplevel(level=[0,1]), 1)
    # DistFeaturesDF['before_average_lastfivegame_duration_trajet'] = DistFeaturesDF.groupby(['id_season', 'tm']).rolling(5)['duration_trajet'].mean().droplevel(level=[0,1])
    # DistFeaturesDF['before_average_lasttengame_duration_trajet'] = DistFeaturesDF.groupby(['id_season', 'tm']).rolling(10)['duration_trajet'].mean().droplevel(level=[0,1])
    # DistFeaturesDF['before_sum_lastfivegame_duration_trajet'] = DistFeaturesDF.groupby(['id_season', 'tm']).rolling(5)['duration_trajet'].sum().droplevel(level=[0,1])
    # DistFeaturesDF['before_sum_lasttengame_duration_trajet'] = DistFeaturesDF.groupby(['id_season', 'tm']).rolling(10)['duration_trajet'].sum().droplevel(level=[0,1])

    ###################
    DistFeaturesDF["before_average_duration_trajet"] = round(
        DistFeaturesDF.groupby(["id_season", "tm"])["duration_trajet"].transform(
            lambda x: x.shift(1).expanding().mean()
        ),
        1,
    )
    DistFeaturesDF["before_average_lastfivegame_duration_trajet"] = round(
        DistFeaturesDF.groupby(["id_season", "tm"])["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(5).mean()
        ),
        1,
    )
    DistFeaturesDF["before_average_lasttengame_duration_trajet"] = round(
        DistFeaturesDF.groupby(["id_season", "tm"])["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(10).mean()
        ),
        1,
    )
    DistFeaturesDF["before_sum_lastfivegame_duration_trajet"] = round(
        DistFeaturesDF.groupby(["id_season", "tm"])["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(5).sum()
        ),
        1,
    )
    DistFeaturesDF["before_sum_lasttengame_duration_trajet"] = round(
        DistFeaturesDF.groupby(["id_season", "tm"])["duration_trajet"].transform(
            lambda x: x.shift(1).rolling(10).sum()
        ),
        1,
    )
    ###################

    # -----------------------------------
    # Fill na values

    DistFeaturesDF["before_average_lastfivegame_duration_trajet"] = DistFeaturesDF[
        "before_average_lastfivegame_duration_trajet"
    ].fillna(DistFeaturesDF["duration_trajet"])
    DistFeaturesDF["before_average_lasttengame_duration_trajet"] = DistFeaturesDF[
        "before_average_lasttengame_duration_trajet"
    ].fillna(DistFeaturesDF["before_average_lastfivegame_duration_trajet"])
    DistFeaturesDF["before_average_lasttengame_duration_trajet"] = DistFeaturesDF[
        "before_average_lasttengame_duration_trajet"
    ].fillna(DistFeaturesDF["duration_trajet"])

    DistFeaturesDF["before_sum_lastfivegame_duration_trajet"] = DistFeaturesDF[
        "before_sum_lastfivegame_duration_trajet"
    ].fillna(DistFeaturesDF["duration_trajet"])
    DistFeaturesDF["before_sum_lasttengame_duration_trajet"] = DistFeaturesDF[
        "before_sum_lasttengame_duration_trajet"
    ].fillna(DistFeaturesDF["before_sum_lastfivegame_duration_trajet"])
    DistFeaturesDF["before_sum_lasttengame_duration_trajet"] = DistFeaturesDF[
        "before_sum_lasttengame_duration_trajet"
    ].fillna(DistFeaturesDF["duration_trajet"])

    # -----------------------------------
    TRAINING_DF = pd.merge(
        TRAINING_DF,
        DistFeaturesDF,
        how="left",
        left_on=["id_season", "game_date", "tm"],
        right_on=["id_season", "game_date", "tm"],
    )

    return TRAINING_DF

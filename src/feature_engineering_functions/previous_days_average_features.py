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
    subset_2 = training_df[["id_season", "game_date", "tm", "duration_trajet"]]

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

    # New method - WORKING
    last_days_features["game_y_n"] = np.where(
        last_days_features["date"] == last_days_features["game_date"], 1, 0
    )

    last_days_features["ext_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"])
        & (last_days_features["extdom"] == "ext"),
        1,
        0,
    )
    last_days_features["dom_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"])
        & (last_days_features["extdom"] == "dom"),
        1,
        0,
    )

    last_days_features["duration_trajet_y_n"] = np.where(
        (last_days_features["date"] == last_days_features["game_date"]),
        last_days_features["duration_trajet"],
        0,
    )

    last_days_features["nb_games_last_5days"] = round(
        last_days_features.groupby(["id_season", "tm"])["game_y_n"].transform(
            lambda x: x.rolling(5).sum()
        ),
        1,
    )
    last_days_features["nb_games_last_7days"] = round(
        last_days_features.groupby(["id_season", "tm"])["game_y_n"].transform(
            lambda x: x.rolling(7).sum()
        ),
        1,
    )
    last_days_features["nb_games_last_10days"] = round(
        last_days_features.groupby(["id_season", "tm"])["game_y_n"].transform(
            lambda x: x.rolling(10).sum()
        ),
        1,
    )

    last_days_features["nb_ext_games_last_5days"] = round(
        last_days_features.groupby(["id_season", "tm"])["ext_y_n"].transform(
            lambda x: x.rolling(5).sum()
        ),
        1,
    )
    last_days_features["nb_ext_games_last_7days"] = round(
        last_days_features.groupby(["id_season", "tm"])["ext_y_n"].transform(
            lambda x: x.rolling(7).sum()
        ),
        1,
    )
    last_days_features["nb_ext_games_last_10days"] = round(
        last_days_features.groupby(["id_season", "tm"])["ext_y_n"].transform(
            lambda x: x.rolling(10).sum()
        ),
        1,
    )

    last_days_features["nb_dom_games_last_5days"] = round(
        last_days_features.groupby(["id_season", "tm"])["dom_y_n"].transform(
            lambda x: x.rolling(5).sum()
        ),
        1,
    )
    last_days_features["nb_dom_games_last_7days"] = round(
        last_days_features.groupby(["id_season", "tm"])["dom_y_n"].transform(
            lambda x: x.rolling(7).sum()
        ),
        1,
    )
    last_days_features["nb_dom_games_last_10days"] = round(
        last_days_features.groupby(["id_season", "tm"])["dom_y_n"].transform(
            lambda x: x.rolling(10).sum()
        ),
        1,
    )

    last_days_features["sum_duration_trajet_y_n_last_5days"] = round(
        last_days_features.groupby(["id_season", "tm"])[
            "duration_trajet_y_n"
        ].transform(lambda x: x.rolling(5).sum()),
        1,
    )
    last_days_features["sum_duration_trajet_y_n_last_7days"] = round(
        last_days_features.groupby(["id_season", "tm"])[
            "duration_trajet_y_n"
        ].transform(lambda x: x.rolling(7).sum()),
        1,
    )
    last_days_features["sum_duration_trajet_y_n_10days"] = round(
        last_days_features.groupby(["id_season", "tm"])[
            "duration_trajet_y_n"
        ].transform(lambda x: x.rolling(10).sum()),
        1,
    )

    # ---------------------------------------------------------------------------------------

    # Only keep the game date
    last_days_features = last_days_features.drop("date", axis=1).drop_duplicates(
        subset=["id_season", "game_nb", "game_date", "extdom", "tm", "opp"],
        keep="first",
    )
    last_days_features = last_days_features.drop(
        ["game_nb", "extdom", "duration_trajet", "game_y_n", "duration_trajet_y_n"],
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

    # -----------------------------------

    training_df["sum_duration_trajet_y_n_last_5days"] = training_df[
        "sum_duration_trajet_y_n_last_5days"
    ].fillna(training_df["duration_trajet"])

    training_df["sum_duration_trajet_y_n_last_7days"] = training_df[
        "sum_duration_trajet_y_n_last_7days"
    ].fillna(training_df["sum_duration_trajet_y_n_last_5days"])
    training_df["sum_duration_trajet_y_n_last_7days"] = training_df[
        "sum_duration_trajet_y_n_last_7days"
    ].fillna(training_df["duration_trajet"])

    training_df["sum_duration_trajet_y_n_10days"] = training_df[
        "sum_duration_trajet_y_n_10days"
    ].fillna(training_df["sum_duration_trajet_y_n_last_7days"])
    training_df["sum_duration_trajet_y_n_10days"] = training_df[
        "sum_duration_trajet_y_n_10days"
    ].fillna(training_df["sum_duration_trajet_y_n_last_5days"])
    training_df["sum_duration_trajet_y_n_10days"] = training_df[
        "sum_duration_trajet_y_n_10days"
    ].fillna(training_df["duration_trajet"])

    # -------------------------------

    training_df["nb_dom_games_last_5days"] = training_df[
        "nb_dom_games_last_5days"
    ].fillna(training_df["dom_y_n"])

    training_df["nb_dom_games_last_7days"] = training_df[
        "nb_dom_games_last_7days"
    ].fillna(training_df["nb_dom_games_last_5days"])
    training_df["nb_dom_games_last_7days"] = training_df[
        "nb_dom_games_last_7days"
    ].fillna(training_df["dom_y_n"])

    training_df["nb_dom_games_last_10days"] = training_df[
        "nb_dom_games_last_10days"
    ].fillna(training_df["nb_dom_games_last_7days"])
    training_df["nb_dom_games_last_10days"] = training_df[
        "nb_dom_games_last_10days"
    ].fillna(training_df["nb_dom_games_last_5days"])
    training_df["nb_dom_games_last_10days"] = training_df[
        "nb_dom_games_last_10days"
    ].fillna(training_df["dom_y_n"])

    # -------------------------------

    training_df["nb_ext_games_last_5days"] = training_df[
        "nb_ext_games_last_5days"
    ].fillna(training_df["ext_y_n"])

    training_df["nb_ext_games_last_7days"] = training_df[
        "nb_ext_games_last_7days"
    ].fillna(training_df["nb_ext_games_last_5days"])
    training_df["nb_ext_games_last_7days"] = training_df[
        "nb_ext_games_last_7days"
    ].fillna(training_df["ext_y_n"])

    training_df["nb_ext_games_last_10days"] = training_df[
        "nb_ext_games_last_10days"
    ].fillna(training_df["nb_ext_games_last_7days"])
    training_df["nb_ext_games_last_10days"] = training_df[
        "nb_ext_games_last_10days"
    ].fillna(training_df["nb_ext_games_last_5days"])
    training_df["nb_ext_games_last_10days"] = training_df[
        "nb_ext_games_last_10days"
    ].fillna(training_df["ext_y_n"])

    # -------------------------------

    training_df["nb_games_last_5days"] = training_df["nb_games_last_5days"].fillna(
        training_df["game_nb"]
    )

    training_df["nb_games_last_7days"] = training_df["nb_games_last_7days"].fillna(
        training_df["nb_games_last_5days"]
    )
    training_df["nb_games_last_7days"] = training_df["nb_games_last_7days"].fillna(
        training_df["game_nb"]
    )

    training_df["nb_games_last_10days"] = training_df["nb_games_last_10days"].fillna(
        training_df["nb_games_last_7days"]
    )
    training_df["nb_games_last_10days"] = training_df["nb_games_last_10days"].fillna(
        training_df["nb_games_last_5days"]
    )
    training_df["nb_games_last_10days"] = training_df["nb_games_last_10days"].fillna(
        training_df["game_nb"]
    )

    training_df = training_df.drop(["Date", "ext_y_n", "dom_y_n"], axis=1)

    return training_df

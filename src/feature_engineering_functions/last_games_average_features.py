import pandas as pd
import numpy as np
import sys

pd.options.mode.chained_assignment = None


def previous_games_average_features(TRAINING_DF):

    TRAINING_DF = TRAINING_DF.sort_values(by=["id_season", "tm", "game_nb"])

    colums_to_process = [
        "pts_tm",
        "pts_opp",
        # 'fg_tm', 'fga_tm', 'fg_prct_tm', '3p_tm', '3pa_tm',
        # '3p_prct_tm', 'ft_tm', 'fta_tm', 'ft_prct_tm', 'orb_tm', 'trb_tm',
        # 'ast_tm', 'stl_tm', 'blk_tm', 'tov_tm', 'pf_tm', 'fg_opp', 'fga_opp',
        # 'fg_prct_opp', '3p_opp', '3pa_opp', '3p_prct_opp', 'ft_opp', 'fta_opp',
        # 'ft_prct_opp', 'orb_opp', 'trb_opp', 'ast_opp', 'stl_opp', 'blk_opp',
        # 'tov_opp', 'pf_opp'
    ]

    for col in colums_to_process:
        # TRAINING_DF['before_average_' + col ] = round(TRAINING_DF.groupby(['id_season', 'tm']).expanding()[col].mean().shift(1).droplevel(level=[0,1]), 1)
        # TRAINING_DF['before_average_lastfivegame_' + col ] = TRAINING_DF.groupby(['id_season', 'tm']).rolling(5)[col].mean().shift(1).droplevel(level=[0,1])
        # TRAINING_DF['before_average_lasttengame_' + col ] = TRAINING_DF.groupby(['id_season', 'tm']).rolling(10)[col].mean().shift(1).droplevel(level=[0,1])

        TRAINING_DF["before_average_" + col] = round(
            TRAINING_DF.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).expanding().mean()
            ),
            1,
        )
        TRAINING_DF["before_average_lastfivegame_" + col] = round(
            TRAINING_DF.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(5).mean()
            ),
            1,
        )
        TRAINING_DF["before_average_lasttengame_" + col] = round(
            TRAINING_DF.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(10).mean()
            ),
            1,
        )

        # Fill na values
        TRAINING_DF["before_average_lastfivegame_" + col] = TRAINING_DF[
            "before_average_lastfivegame_" + col
        ].fillna(TRAINING_DF["before_average_" + col])
        TRAINING_DF["before_average_lasttengame_" + col] = TRAINING_DF[
            "before_average_lasttengame_" + col
        ].fillna(TRAINING_DF["before_average_lastfivegame_" + col])
        TRAINING_DF["before_average_lasttengame_" + col] = TRAINING_DF[
            "before_average_lasttengame_" + col
        ].fillna(TRAINING_DF["before_average_" + col])

        TRAINING_DF["diff_all_minus_lastfivegame_" + col] = (
            TRAINING_DF["before_average_" + col]
            - TRAINING_DF["before_average_lastfivegame_" + col]
        )
        TRAINING_DF["diff_all_minus_lasttengame_" + col] = (
            TRAINING_DF["before_average_" + col]
            - TRAINING_DF["before_average_lasttengame_" + col]
        )

    return TRAINING_DF


def previous_games_ratio_average_features(TRAINING_DF):

    df = TRAINING_DF[
        ["id_season", "game_nb", "game_date", "extdom", "tm", "opp", "results"]
    ]
    df["w_for_ratio"] = np.where(df["results"] == "W", 1, 0)
    df["cum_sum_win"] = df.groupby(["id_season", "tm"])["w_for_ratio"].cumsum()
    df["average_ratio"] = df["cum_sum_win"] / df["game_nb"]

    df["before_average_W_ratio"] = df.groupby(["id_season", "tm"])[
        "average_ratio"
    ].shift(1)

    ###################

    df_ratio_last_5_games = (
        df.set_index("game_date")
        .groupby(["id_season", "tm"])
        .rolling(5)["w_for_ratio"]
        .sum()
        .reset_index()
    )
    df_ratio_last_5_games["before_average_lastfivegame_W_ratio"] = (
        df_ratio_last_5_games["w_for_ratio"] / 5
    )
    df_ratio_last_5_games[
        "before_average_lastfivegame_W_ratio"
    ] = df_ratio_last_5_games.groupby(["id_season", "tm"])[
        "before_average_lastfivegame_W_ratio"
    ].shift(
        1
    )

    df_ratio_last_10_games = (
        df.set_index("game_date")
        .groupby(["id_season", "tm"])
        .rolling(10)["w_for_ratio"]
        .sum()
        .reset_index()
    )
    df_ratio_last_10_games["before_average_lasttengame_W_ratio"] = (
        df_ratio_last_10_games["w_for_ratio"] / 10
    )
    df_ratio_last_10_games[
        "before_average_lasttengame_W_ratio"
    ] = df_ratio_last_10_games.groupby(["id_season", "tm"])[
        "before_average_lasttengame_W_ratio"
    ].shift(
        1
    )

    df_ratio_last = pd.merge(
        df_ratio_last_5_games,
        df_ratio_last_10_games,
        how="left",
        left_on=["id_season", "game_date", "tm"],
        right_on=["id_season", "game_date", "tm"],
    )

    final = pd.merge(
        df,
        df_ratio_last,
        how="left",
        left_on=["id_season", "game_date", "tm"],
        right_on=["id_season", "game_date", "tm"],
    )

    # Fill na values
    final["before_average_lastfivegame_W_ratio"] = final[
        "before_average_lastfivegame_W_ratio"
    ].fillna(final["before_average_W_ratio"])
    final["before_average_lasttengame_W_ratio"] = final[
        "before_average_lasttengame_W_ratio"
    ].fillna(final["before_average_lastfivegame_W_ratio"])
    final["before_average_lasttengame_W_ratio"] = final[
        "before_average_lasttengame_W_ratio"
    ].fillna(final["before_average_W_ratio"])

    final["diff_all_minus_lastfivegame_W_ratio"] = (
        final["before_average_W_ratio"] - final["before_average_lastfivegame_W_ratio"]
    )
    final["diff_all_minus_lasttengame_W_ratio"] = (
        final["before_average_W_ratio"] - final["before_average_lasttengame_W_ratio"]
    )

    final = final.drop(
        [
            "game_nb",
            "extdom",
            "results",
            "w_for_ratio",
            "cum_sum_win",
            "average_ratio",
            "w_for_ratio_x",
            "w_for_ratio_y",
        ],
        axis=1,
    )

    TRAINING_DF = pd.merge(
        TRAINING_DF,
        final,
        how="left",
        left_on=["id_season", "game_date", "tm", "opp"],
        right_on=["id_season", "game_date", "tm", "opp"],
    )

    return TRAINING_DF


def previous_season_ratio_features(TRAINING_DF):

    MaxGameID = pd.DataFrame(
        TRAINING_DF.groupby(["id_season", "tm"])["game_nb"].max()
    ).reset_index()
    MaxGameID.columns = ["id_season", "tm", "max_game_id"]

    TRAINING_DF = pd.merge(
        TRAINING_DF,
        MaxGameID,
        how="left",
        left_on=["id_season", "tm"],
        right_on=["id_season", "tm"],
    )

    LastSeasonRatioFeatures_df = TRAINING_DF[
        TRAINING_DF["game_nb"] == TRAINING_DF["max_game_id"]
    ]

    LastSeasonRatioFeatures_df["before_season_ratio"] = (
        LastSeasonRatioFeatures_df["w_tot"] / LastSeasonRatioFeatures_df["max_game_id"]
    )
    LastSeasonRatioFeatures_df["id_season_plus1"] = (
        LastSeasonRatioFeatures_df["id_season"] + 1
    )

    LastSeasonRatioFeatures_df = LastSeasonRatioFeatures_df[
        ["id_season_plus1", "tm", "before_season_ratio"]
    ]

    TRAINING_DF = pd.merge(
        TRAINING_DF,
        LastSeasonRatioFeatures_df,
        how="left",
        left_on=["id_season", "tm"],
        right_on=["id_season_plus1", "tm"],
    )

    TRAINING_DF = TRAINING_DF.drop(["id_season_plus1", "max_game_id"], axis=1)
    TRAINING_DF["before_season_ratio"] = TRAINING_DF["before_season_ratio"].fillna(0)
    return TRAINING_DF

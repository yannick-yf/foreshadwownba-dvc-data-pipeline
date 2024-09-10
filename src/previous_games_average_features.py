import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np
from typing import Text
import yaml
import argparse
from src.utils.logs import get_logger


def previous_games_average_features(config_path: Text) -> pd.DataFrame:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """
    with open("params.yaml") as conf_file:
        config_params = yaml.safe_load(conf_file)

    logger = get_logger(
        "PREVIOUS_GAMES_AVG_FEATURES", log_level=config_params["base"]["log_level"]
    )

    # Read the input data for the step
    training_dataset = pd.read_csv(
        "./data/processed/nba_games_training_dataset_pre_cleaned.csv"
    )

    training_dataset = training_dataset.sort_values(by=["id_season", "tm", "game_nb"])

    colums_to_process = [
        "pts_tm",
        "pts_opp",
        "fg_tm",
        "fga_tm",
        "fg_prct_tm",
        "3p_tm",
        "3pa_tm",
        "3p_prct_tm",
        "ft_tm",
        "fta_tm",
        "ft_prct_tm",
        "orb_tm",
        "trb_tm",
        "ast_tm",
        "stl_tm",
        "blk_tm",
        "tov_tm",
        "pf_tm",
        "fg_opp",
        "fga_opp",
        "fg_prct_opp",
        "3p_opp",
        "3pa_opp",
        "3p_prct_opp",
        "ft_opp",
        "fta_opp",
        "ft_prct_opp",
        "orb_opp",
        "trb_opp",
        "ast_opp",
        "stl_opp",
        "blk_opp",
        "tov_opp",
        "pf_opp",
    ]

    new_training_dataset = training_dataset.copy()

    for col in colums_to_process:
        new_training_dataset["before_average_" + col] = round(
            new_training_dataset.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).expanding().mean()
            ),
            1,
        )
        new_training_dataset["before_average_lastfivegame_" + col] = round(
            new_training_dataset.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(5).mean()
            ),
            1,
        )
        new_training_dataset["before_average_lasttengame_" + col] = round(
            new_training_dataset.groupby(["id_season", "tm"])[col].transform(
                lambda x: x.shift(1).rolling(10).mean()
            ),
            1,
        )

        # Fill na values
        new_training_dataset["before_average_lastfivegame_" + col] = new_training_dataset[
            "before_average_lastfivegame_" + col
        ].fillna(new_training_dataset["before_average_" + col])
        new_training_dataset["before_average_lasttengame_" + col] = new_training_dataset[
            "before_average_lasttengame_" + col
        ].fillna(new_training_dataset["before_average_lastfivegame_" + col])
        new_training_dataset["before_average_lasttengame_" + col] = new_training_dataset[
            "before_average_lasttengame_" + col
        ].fillna(new_training_dataset["before_average_" + col])

        new_training_dataset["diff_all_minus_lastfivegame_" + col] = (
            new_training_dataset["before_average_" + col]
            - new_training_dataset["before_average_lastfivegame_" + col]
        )
        new_training_dataset["diff_all_minus_lasttengame_" + col] = (
            new_training_dataset["before_average_" + col]
            - new_training_dataset["before_average_lasttengame_" + col]
        )

    # Save the data
    new_training_dataset.to_csv(
        "./data/processed/nba_games_training_dataset_previous_games_features.csv",
        index=False,
    )

    logger.info("Previous Games Average Features step complete")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config-params", dest="config_params", required=True)

    args = arg_parser.parse_args()

    previous_games_average_features(config_path=args.config_params)

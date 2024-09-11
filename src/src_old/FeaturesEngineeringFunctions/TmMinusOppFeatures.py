# TmMinusOppFeatures.py

import pandas as pd
import numpy as np

def TmMinusOppFeatures(TrainingSet):

    #---------------
    # pts average vs points encaisse de lequipe adverse
    TrainingSet['before_average_pts_tm_x_Minus_pts_opp_y'] = TrainingSet['before_average_pts_tm_x'] - TrainingSet['before_average_pts_opp_y']
    TrainingSet['before_average_pts_opp_x_Minus_pts_tm_y'] = TrainingSet['before_average_pts_opp_x'] - TrainingSet['before_average_pts_tm_y']

    TrainingSet['before_average_lastfivegame_pts_tm_x_Minus_pts_opp_y'] = TrainingSet['before_average_lastfivegame_pts_tm_x'] - TrainingSet['before_average_lastfivegame_pts_opp_y']
    TrainingSet['before_average_lastfivegame_pts_opp_x_Minus_pts_tm_y'] = TrainingSet['before_average_lastfivegame_pts_opp_x'] - TrainingSet['before_average_lastfivegame_pts_tm_y']

    TrainingSet['before_average_lasttengame_pts_tm_x_Minus_pts_opp_y'] = TrainingSet['before_average_lasttengame_pts_tm_x'] - TrainingSet['before_average_lasttengame_pts_opp_y']
    TrainingSet['before_average_lasttengame_pts_opp_x_Minus_pts_tm_y'] = TrainingSet['before_average_lasttengame_pts_opp_x'] - TrainingSet['before_average_lasttengame_pts_tm_y']

    #---------------
    # Steal vs turn over tm vs opponent
    TrainingSet['before_average_stl_tm_x_Minus_tov_tm_y'] = TrainingSet['before_average_stl_tm_x'] - TrainingSet['before_average_tov_tm_y']
    TrainingSet['before_average_tov_tm_x_Minus_tl_tm_y'] = TrainingSet['before_average_tov_tm_x'] - TrainingSet['before_average_stl_tm_y']

    TrainingSet['before_average_lastfivegame_stl_tm_x_Minus_tov_tm_y'] = TrainingSet['before_average_lastfivegame_stl_tm_x'] - TrainingSet['before_average_lastfivegame_tov_tm_y']
    TrainingSet['before_average_lastfivegame_tov_tm_x_Minus_tl_tm_y'] = TrainingSet['before_average_lastfivegame_tov_tm_x'] - TrainingSet['before_average_lastfivegame_stl_tm_y']

    TrainingSet['before_average_lasttengame_stl_tm_x_Minus_tov_tm_y'] = TrainingSet['before_average_lasttengame_stl_tm_x'] - TrainingSet['before_average_lasttengame_tov_tm_y']
    TrainingSet['before_average_lasttengame_tov_tm_x_Minus_tl_tm_y'] = TrainingSet['before_average_lasttengame_tov_tm_x'] - TrainingSet['before_average_lasttengame_stl_tm_y']

    #-----------------
    # 3point scored vs 3pont encaisse
    TrainingSet['before_average_3p_tm_x_Minus_3p_opp_y'] = TrainingSet['before_average_3p_tm_x'] - TrainingSet['before_average_3p_opp_y']
    TrainingSet['before_average_3p_tm_y_Minus_3p_opp_x'] = TrainingSet['before_average_3p_tm_y'] - TrainingSet['before_average_3p_opp_x']

    TrainingSet['before_average_lastfivegame_3p_tm_x_Minus_3p_opp_y'] = TrainingSet['before_average_lastfivegame_3p_tm_x'] - TrainingSet['before_average_lastfivegame_3p_opp_y']
    TrainingSet['before_average_lastfivegame_3p_tm_y_Minus_3p_opp_x'] = TrainingSet['before_average_lastfivegame_3p_tm_y'] - TrainingSet['before_average_lastfivegame_3p_opp_x']

    TrainingSet['before_average_lasttengame_3p_tm_x_Minus_3p_opp_y'] = TrainingSet['before_average_lasttengame_3p_tm_x'] - TrainingSet['before_average_lasttengame_3p_opp_y']
    TrainingSet['before_average_lasttengame_3p_tm_y_Minus_3p_opp_x'] = TrainingSet['before_average_lasttengame_3p_tm_y'] - TrainingSet['before_average_lasttengame_3p_opp_x']

    return TrainingSet

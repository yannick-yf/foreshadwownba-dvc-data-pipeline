# fn_get_y_variable.py

import pandas as pd
import numpy as np

def best_team_name(row):
    if (row['before_average_W_ratio_x'] > row['before_average_W_ratio_y']) :
        val = row['tm_x']
    elif (row['before_average_W_ratio_x'] < row['before_average_W_ratio_y']):
        val = row['opp_x']
    else:
        if (row['before_average_lasttengame_W_ratio_x'] > row['before_average_lasttengame_W_ratio_y']):
            val = row['tm_x']
        elif (row['before_average_lasttengame_W_ratio_x'] < row['before_average_lasttengame_W_ratio_y']):
            val = row['opp_x']
        else:
            if (row['before_average_lastfivegame_W_ratio_x'] > row['before_average_lastfivegame_W_ratio_y']):
                val = row['tm_x']
            elif (row['before_average_lastfivegame_W_ratio_x'] < row['before_average_lastfivegame_W_ratio_y']):
                val = row['opp_x']
            else:
                if (row['before_streak_w_l_x'] > row['before_streak_w_l_y']):
                    val = row['tm_x']
                elif (row['before_streak_w_l_x'] < row['before_streak_w_l_y']):
                    val = row['opp_x']
                else:
                    if (row['before_season_ratio_x'] > row['before_season_ratio_y']):
                        val = row['tm_x']
                    elif (row['before_season_ratio_x'] < row['before_season_ratio_y']):
                        val = row['opp_x']
                    else:
                        if (row['rest_x'] > row['rest_y']):
                            val = row['tm_x']
                        elif (row['rest_x'] < row['rest_y']):
                            val = row['opp_x']
                        else:
                            if row['extdom_x'] == 'dom':
                                val = row['tm_x']
                            else:
                                val = row['opp_x']
    return val

def fn_get_y_variables(training_df):

    training_df['results'] = np.where(
        training_df['pts_tm_x'] > training_df['pts_opp_x'],
        'W',
        'L')

    training_df['results'] = training_df['results'].astype(str)
    training_df['name_best_team'] = training_df.apply(best_team_name, axis=1)

    training_df['results'] = np.where(
        training_df['pts_tm_x'] > training_df['pts_opp_x'],
        1,
        0)

    training_df['y_prob_win'] = np.where(
        training_df['results'] == 1,
        'W',
        'L')

    training_df['y_prob_win'] = np.where(
        training_df['y_prob_win'] == 'W',
        '1',
        '0')

    training_df['name_win_team'] = np.where(
        training_df['pts_tm_x'] > training_df['pts_opp_x'],
        training_df['tm_x'], training_df['opp_x'])

    training_df['y_bestworst'] = np.where(
        training_df['name_best_team'] == training_df['name_win_team'],
        1,
        0)

    return training_df


def fn_get_unique_id(training_df):

    # training_df['id_1'] = np.where(
    #     training_df['extdom_x']=='dom',
    #     training_df['game_date'] + '_' + training_df['opp_x'] + '_' + training_df['tm_x'],
    #     training_df['game_date'] + '_' + training_df['tm_x'] + '_' + training_df['opp_x']
    # )

    training_df['id'] = np.where(
        training_df['extdom_x']=='dom',
        training_df['game_date'].astype(str) + '_' + training_df['opp_x'] + '_' + training_df['tm_x'] + '_win' + training_df['name_win_team'] + '_best' + training_df['name_best_team'],
        training_df['game_date'].astype(str)  + '_' + training_df['tm_x'] + '_' + training_df['opp_x'] + '_win' + training_df['name_win_team'] + '_best' + training_df['name_best_team']
    )

    return training_df


def get_final_bestworst_training_dataset(training_df):

    # Rename columns with _x : df.columns = df.columns.str.replace("[()]", "_")
    subset1 = training_df[training_df['name_best_team'] == training_df['tm_x']]

    subset1.columns = subset1.columns.str.replace(r'_x$', "_best", regex=True) # r'[./-]$
    subset1.columns = subset1.columns.str.replace(r'_y$', "_worst", regex=True)

    ids_cols = ['id', 'y_bestworst', 'id_season_best', 'game_date', 'name_best_team']
    best_cols = [col for col in subset1 if col.endswith('_best')]
    best_cols.remove('id_season_best')

    worst_cols = [col for col in subset1 if col.endswith('_worst')]
    worst_cols.remove('id_season_worst')

    columns_list = ids_cols + best_cols + worst_cols

    subset1 = subset1[columns_list]

    ######

    subset2 = training_df[training_df['name_best_team'] == training_df['opp_x']]

    subset2.columns = subset2.columns.str.replace(r'_y$', "_best", regex=True) # r'[./-]$
    subset2.columns = subset2.columns.str.replace(r'_x$', "_worst", regex=True)

    ids_cols = ['id', 'y_bestworst', 'id_season_best', 'game_date', 'name_best_team']
    best_cols = [col for col in subset2 if col.endswith('_best')]
    best_cols.remove('id_season_best')

    worst_cols = [col for col in subset2 if col.endswith('_worst')]
    worst_cols.remove('id_season_worst')

    columns_list = ids_cols + best_cols + worst_cols

    subset2 = subset2[columns_list]

    #######

    FinalTraining = pd.concat([subset1, subset2], axis = 0)
    return FinalTraining

def fn_get_one_line_per_games(FinalTraining):
    
    FinalTraining = FinalTraining.drop_duplicates()
    print('Number of rows -  1 lines per game: ' + str(FinalTraining.shape[0]))
    return FinalTraining

def fn_FeaturesEnginnering_part2(FinalTraining):

    # Best - Worst Features exemple : points encaissés par x - points marsquer par y
    FinalTraining['before_average_ratio_diff_bestworst'] = FinalTraining['before_average_W_ratio_best'] - FinalTraining['before_average_W_ratio_worst']

    # Catege recoding
    FinalTraining['extdom_best'] = np.where(FinalTraining['extdom_best']=='dom', 1, 0)

    return FinalTraining



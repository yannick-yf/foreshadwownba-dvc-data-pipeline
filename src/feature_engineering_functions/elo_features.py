
# EloFeatures.py
# Method come from the following script:
# https://nbviewer.org/github/practicallypredictable/posts/blob/master/basketball/nba/notebooks/nba-simple_elo_ratings.ipynb

# ELO
# https://www.ergosum.co/nate-silvers-nba-elo-algorithm/
# https://github.com/rogerfitz/tutorials/blob/master/Nate%20Silver%20ELO/NBA%20ELO%20Replicate.ipynb
# https://fivethirtyeight.datasettes.com/fivethirtyeight/nba-elo%2Fnbaallelo
# https://harvardsportsanalysis.org/2019/01/a-simple-improvement-to-fivethirtyeights-nba-elo-model/
# https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings - IMPORTANT ONE -Wloc = extr/dom : H = home | A = Away
# https://nbviewer.org/github/practicallypredictable/posts/blob/master/basketball/nba/notebooks/nba-simple_elo_ratings.ipynb - VERY IMPORTANT


import numpy as np
import pandas as pd
# from sklearn.metrics import log_loss
import sys
import math
from collections import OrderedDict
from enum import Enum

#--------------------------------------------------
# Elo Functions

def win_probs(*, home_elo, road_elo, hca_elo):
    """Home and road team win probabilities implied by Elo ratings and home court adjustment."""
    h = math.pow(10, home_elo/400)
    r = math.pow(10, road_elo/400)
    a = math.pow(10, hca_elo/400)
    denom = r + a*h
    home_prob = a*h / denom
    road_prob = r / denom
    return home_prob, road_prob

###

def home_odds_on(*, home_elo, road_elo, hca_elo):
    """Odds in favor of home team implied by Elo ratings and home court adjustment."""
    h = math.pow(10, home_elo/400)
    r = math.pow(10, road_elo/400)
    a = math.pow(10, hca_elo/400)
    return a*h/r

###

def hca_calibrate(*, home_win_prob):
    """Calibrate Elo home court adjustment to a given historical home team win percentage."""
    if home_win_prob <= 0 or home_win_prob >= 1:
        raise ValueError('invalid home win probability', home_win_prob)
    a = home_win_prob / (1 - home_win_prob)
    print(f'a = {a}')
    hca = 400 * math.log10(a)
    return hca


###

def update(*, winner, home_elo, road_elo, hca_elo, k, probs=False):
    """Update Elo ratings for a given match up."""
    home_prob, road_prob = win_probs(home_elo=home_elo, road_elo=road_elo, hca_elo=hca_elo)
    if winner[0].upper() == 'H':
        home_win = 1
        road_win = 0
    elif winner[0].upper() in ['R', 'A', 'V']: # road, away or visitor are treated as synonyms
        home_win = 0
        road_win = 1
    else:
        raise ValueError('unrecognized winner string', winner)
    new_home_elo = home_elo + k*(home_win - home_prob)
    new_road_elo = road_elo + k*(road_win - road_prob)
    if probs:
        return new_home_elo, new_road_elo, home_prob, road_prob
    else:
        return new_home_elo, new_road_elo

###

def simple_nba_elo(*, box_scores, teams, hca_elo, k):
    """Compute simple Elo ratings over the course of an NBA season."""
    latest_elos = {abbr: 1500 for abbr in teams['abbr']}
    #matchups = box_scores.matchups.sort_values(by='date', ascending=True).copy()
    matchups = box_scores.sort_values(by='date', ascending=True).copy()
    home_probs = []
    road_probs = []
    home_elos = []
    road_elos = []
    index_check = []
    elo_ts = []
    for game in matchups.itertuples(index=True):
        index = game.Index
        home_team = game.team_abbr_h
        road_team = game.team_abbr_r
        winner = game.hr_winner
        home_elo = latest_elos[home_team]
        road_elo = latest_elos[road_team]
        (new_home_elo, new_road_elo, home_prob, road_prob) = update(
            winner=winner,
            home_elo=home_elo,
            road_elo=road_elo,
            hca_elo=hca_elo,
            k=k,
            probs=True
        )
        home_info = OrderedDict({
            'date': game.date,
            'game_id': game.game_id,
            'abbr': home_team,
            'matchup_index': index,
            'opp_abbr': road_team,
            'home_road': 'H',
            'win_loss': 'W' if winner == 'H' else 'L',
            'win_prob': home_prob,
            'opp_prior_elo': latest_elos[road_team],
            'prior_elo': latest_elos[home_team],
            'new_elo': new_home_elo,
        })
        elo_ts.append(home_info)
        road_info = OrderedDict({
            'date': game.date,
            'game_id': game.game_id,
            'abbr': road_team,
            'matchup_index': index,
            'opp_abbr': home_team,
            'home_road': 'R',
            'win_loss': 'W' if winner == 'R' else 'L',
            'win_prob': road_prob,
            'opp_prior_elo': latest_elos[home_team],
            'prior_elo': latest_elos[road_team],
            'new_elo': new_road_elo,
        })
        elo_ts.append(road_info)
        latest_elos[home_team] = new_home_elo
        latest_elos[road_team] = new_road_elo
        home_probs.append(home_prob)
        road_probs.append(road_prob)
        home_elos.append(new_home_elo)
        road_elos.append(new_road_elo)
        index_check.append(index)
    matchups['home_prob'] = home_probs
    matchups['road_prob'] = road_probs
    matchups['home_elos'] = home_elos
    matchups['road_elos'] = road_elos
    matchups['index_check'] = index_check
    if not all(matchups['index_check'] == matchups.index):
        raise RuntimeError('indices do not match!')
    matchups = matchups.drop(columns=['index_check'])
    return matchups, pd.DataFrame(elo_ts), latest_elos



def elo_features_pipeline(TRAINING_DF):

    #--------------------------------------------------
    # Data Process Before Elo Execution

    rs = TRAINING_DF[['id_season', 'game_date', 'extdom', 'tm', 'opp', 'pts_tm', 'pts_opp']]

    rs_final =  pd.merge(
        rs.copy(),
        rs.copy(),
        how='left',
        left_on=['game_date', 'tm'],
        right_on=['game_date', 'opp'])


    rs_final['team_abbr_h'] = np.where(rs_final['extdom_x']=='dom', rs_final['tm_x'], rs_final['opp_x'])
    rs_final['team_abbr_r'] = np.where(rs_final['extdom_x']=='dom', rs_final['opp_x'], rs_final['tm_x'])

    rs_final['pts_h'] = np.where(rs_final['extdom_x']=='dom', rs_final['pts_tm_x'], rs_final['pts_opp_x'])
    rs_final['pts_r'] = np.where(rs_final['extdom_x']=='dom', rs_final['pts_opp_x'], rs_final['pts_tm_x'])

    rs_final['win_loss_h'] = np.where(
        (rs_final['extdom_x']=='dom') & (rs_final['pts_tm_x'] > rs_final['pts_opp_x']) | (rs_final['extdom_x']=='ext') & (rs_final['pts_tm_x'] < rs_final['pts_opp_x']) ,
        'W',
        'L')

    rs_final['win_loss_r'] = np.where(rs_final['win_loss_h']=='W', 'L', 'W')

    rs_final['winner'] = np.where(
        rs_final['win_loss_h']=='W',
        rs_final['team_abbr_h'],
        rs_final['team_abbr_r'])

    rs_final['loser'] = np.where(
        rs_final['win_loss_h']=='L',
        rs_final['team_abbr_h'],
        rs_final['team_abbr_r'])

    rs_final['date'] = rs_final['game_date']
    rs_final['season'] = rs_final['id_season_x']

    rs_final['hr_winner'] = np.where(rs_final['win_loss_h']=='W', 'H', 'R')

    rs_final['game_id'] = rs_final['date'].astype(str) + rs_final['team_abbr_h'].astype(str)  + rs_final['team_abbr_r'].astype(str) 

    rs_final = rs_final[['game_id', 'season', 'date', 'team_abbr_h', 'team_abbr_r', 'win_loss_h', 'win_loss_r', 'winner', 'loser', 'pts_h', 'pts_r', 'hr_winner']]
    rs_final = rs_final.drop_duplicates()

    #--------------------------------------------------
    # Elo Varaibles Set up

    x = np.linspace(-1200, 1200, 240)

    hca_elo = hca_calibrate(home_win_prob=0.598)
    # vec_win_probs = np.vectorize(win_probs)
    # hca_y = vec_win_probs(home_elo=x, road_elo=0, hca_elo=hca_elo)[0]

    teams = pd.DataFrame()
    teams['abbr'] = rs_final.team_abbr_h.unique()

    #--------------------------------------------------
    # EloExecution

    matchups, elo_hist, curr_elos = simple_nba_elo(
        box_scores=rs_final,
        teams=teams,
        hca_elo=hca_elo,
        k=20)

    #--------------------------------------------------
    # Reformating before adding Elofeatures

    elo_hist = elo_hist[[
        'date',
        'abbr',
        #'opp_abbr',
        'win_prob',
        'opp_prior_elo',
        'prior_elo']]

    elo_hist.rename({
        'date': 'game_date',
        'abbr': 'tm',
        #'opp_abbr': 'opp',
        'win_prob': 'tm_win_prob_elo',
        'prior_elo': 'tm_prior_elo',
        }, axis=1, inplace=True)

    #--------------------------------------------------
    # Add EloFeatures to the Training dataset

    TRAINING_DF =  pd.merge(
        TRAINING_DF,
        elo_hist,
        how='left',
        left_on=['game_date', 'tm'],
        right_on=['game_date', 'tm'])

    return TRAINING_DF


### Alternative Method ###

# import numpy as np
# import pandas as pd
# from sklearn.metrics import log_loss
# import sys

# # Season	DayNum	WTeamID	WScore	LTeamID	LScore	WLoc	NumOT
# #   1985	    20	   1228	    81	   1328	    64	   N	    0
# #   1985	    25	   1106	    77	   1354	    70	   H	    0
# #   1985	    25	   1112	    63	   1223	    56	   H	    0

# K = 20.
# HOME_ADVANTAGE = 60

# def elo_pred(elo1, elo2):
#     return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

# def expected_margin(elo_diff):
#     return((7.5 + 0.006 * elo_diff))

# def elo_update(w_elo, l_elo, margin):
#     elo_diff = w_elo - l_elo
#     pred = elo_pred(w_elo, l_elo)
#     mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
#     update = K * mult * (1 - pred)
#     return(pred, update)

# #--------------------------------------------------

# rs = pd.read_csv('./pipeline_output/final_training_dataset_2022-01-20.csv') 

# rs['overtime'] = rs['overtime'].fillna('NOT')

# # print(rs.columns)
# rs = rs[['id_season', 'game_nb', 'game_date', 'extdom', 'tm', 'opp', 'pts_tm', 'pts_opp']]

# # ext = -1
# # dom = 1
# rs['extdom'] = np.where(rs['extdom']=='@', -1, 1)

# # if tm win game so extdom = 1 and WLoc =1
# # if tm lost game, that means opp win the game, so opposite of extdom tm is the good value
# # Because 1 mean dom and -1 equal to away.
# # by just mulitply by -1 we got the value needed if opp win the game 
# rs['WLoc'] = np.where(
#     rs['pts_tm']>rs['pts_opp'],
#     rs['extdom'],
#     rs['extdom']*-1)

# rs['WLoc'] = np.where(rs['WLoc']==-1,'A','H')

# rs['WTeamID'] = np.where(
#     rs['pts_tm']>rs['pts_opp'],
#     rs['tm'],
#     rs['opp'])

# rs['WScore'] = np.where(
#     rs['pts_tm']>rs['pts_opp'],
#     rs['pts_tm'],
#     rs['pts_opp'])

# rs['LTeamID'] = np.where(
#     rs['pts_tm']>rs['pts_opp'],
#     rs['opp'],
#     rs['tm'])

# rs['LScore'] = np.where(
#     rs['pts_tm']>rs['pts_opp'],
#     rs['pts_opp'],
#     rs['pts_tm'])

# rs['Season'] = rs['id_season']

# rs = rs[['game_date', 'Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc']]
# rs = rs.drop_duplicates().copy()
# rs = rs.reset_index(drop=True)
# team_ids = set(rs.WTeamID).union(set(rs.LTeamID))

# # This dictionary will be used as a lookup for current
# # scores while the algorithm is iterating through each game
# elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

# # This dictionary will be used as a lookup for current
# # scores while the algorithm is iterating through each game
# elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

# # Elo updates will be scaled based on the margin of victory
# rs['margin'] = rs.WScore - rs.LScore

# # I'm going to iterate over the games dataframe using 
# # index numbers, so want to check that nothing is out
# # of order before I do that.
# assert np.all(rs.index.values == np.array(range(rs.shape[0]))), "Index is out of order."

# #-----------------------------------------------
# # Function Execution

# preds = []
# w_elo = []
# l_elo = []

# # Loop over all rows of the games dataframe
# for row in rs.itertuples():
    
#     # Get key data from current row
#     w = row.WTeamID
#     l = row.LTeamID
#     margin = row.margin
#     wloc = row.WLoc
    
#     # Does either team get a home-court advantage?
#     w_ad, l_ad, = 0., 0.
#     if wloc == "H":
#         w_ad += HOME_ADVANTAGE
#     elif wloc == "A":
#         l_ad += HOME_ADVANTAGE
    
#     # Get elo updates as a result of the game
#     pred, update = elo_update(
#         elo_dict[w] + w_ad,
#         elo_dict[l] + l_ad, 
#         margin)

#     elo_dict[w] += update
#     elo_dict[l] -= update
    
#     # Save prediction and new Elos for each round
#     preds.append(pred)
#     w_elo.append(elo_dict[w])
#     l_elo.append(elo_dict[l])

# rs['w_elo'] = w_elo
# rs['l_elo'] = l_elo

# test = rs[ (rs['Season']==2018) & ((rs['WTeamID'] == 'NYK') | (rs['LTeamID'] == 'NYK'))]

# print(test.sort_values(by='game_date'))

# # print(np.mean(-np.log(preds)))
# sys.exit()

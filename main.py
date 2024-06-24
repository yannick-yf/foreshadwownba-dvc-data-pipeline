# main.py

import pandas as pd
import os
# TODO: Auto ML script under development using XgboostNormalSplit2LGStartSeason.ipynb notebooks
# TODO: Grid Search
# TODO: script to save results on kfold and on the test set
# TODO: TRy stacking and ensemble methods
# TODO: Append all the results on only one file that we will send to Antho/Flo


# odds web site: https://www.coteur.com/calculette-surebet.php
# https://www.kaggle.com/erichqiu/nba-odds-and-scores

#--------------------------------------------
# Features engineering pipeline
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3827474

#------------------------------------------------------------------
# Features and models selection according the proba of a given models to maximize the accuracy of the models
# https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
# https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/
# https://machinelearningmastery.com/dynamic-classifier-selection-in-python/
# https://machinelearningmastery.com/feature-selection-with-optimization/
# https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/
# https://machinelearningmastery.com/super-learner-ensemble-in-python/
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/#:~:text=A%20machine%20learning%20pipeline%20can,regression%2C%20and%20post%2Dprocessing.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV
#--------------------------------------------
# Resaerch paper
# https://community.wolfram.com/groups/-/m/t/1730466
# http://dionny.github.io/NBAPredictions/website/

#--------------------------------------------
# Grid Search code python with xgboost as it was done with R
# https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

#------------------------------------------------------
# MODEL
# https://arxiv.org/pdf/1809.06705.pdf - https://github.com/Liam-E2/RotationForest - Rotation Radnom forest
# https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/ - Stacking methods
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ - SMOTE Method
# https://towardsdatascience.com/practical-guide-to-ensemble-learning-d34c74e022a0
# https://christophm.github.io/interpretable-ml-book/rulefit.html#advantages-4
# https://github.com/christophM/rulefit
# https://towardsdatascience.com/interpretable-machine-learning-in-10-minutes-with-rulefit-and-scikit-learn-da9ebb925795
# TODO : VERY Important : https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html
# https://github.com/csinva/imodels/blob/master/notebooks/imodels_demo.ipynb

#--------------------------------------------
# Optimizing Bet
# https://www.kdnuggets.com/2019/06/optimization-python-money-risk.html

# https://docs.python.org/3/howto/logging.html

# New feauters for models
# https://www.basketball-reference.com/leagues/NBA_2019_standings_by_date_eastern_conference.html

# Bayesian grid search
# https://towardsdatascience.com/using-pandas-pipe-function-to-improve-code-readability-96d66abfaf8
# Bayesian grid search to implem,ent

# Error Analysis
# https://pypi.org/project/raiwidgets/
# https://erroranalysis.ai/raiwidgets.html
# https://erroranalysis.ai/
# https://github.com/microsoft/responsible-ai-toolbox/tree/main/erroranalysis

import sys
sys.path.append(".")

from functions.fn_get_training_dataset import fn_get_training_dataset
from functions.LastExtraProcess import fn_get_y_variables, fn_get_unique_id, get_final_bestworst_training_dataset, fn_get_one_line_per_games, fn_FeaturesEnginnering_part2
from functions.GetTrainingDatasetBestWorst import GetTrainingDatasetBestWorst
from functions.GetTrainingDataset2LinePerGames import GetTrainingDataset2LinePerGames
from src_old.FeaturesEngineering import FeaturesEngineeringPipeline
from functions.CleaningData import CleaningData 
from functions.MachineLearning import MachineLearning
# from functions.BusinessRules import BusinessRules
from functions.PathCreation import PathCreation

# from functions. import StackingModels
# from functions.OddsFeatures import AddOddsValues

import logging

# logging.getLogger('matplotlib.font_manager').disabled = True

# Create logger and assign handler
logger = logging.getLogger("Model_Execution")
handler  = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger("Model_Execution.iter")
logger.setLevel(logging.INFO)

import datetime

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Data verification function - Out of the process
    # need to check that for each season each teams got the same number of games that a pre-requistes for the pipeline
    # if it is not the case return warning message with id season / teams

    start_time = datetime.datetime.now()

    # #------------------------------------------------------------------
    # Get training dataset with one row per team (2rows per games)
    training_dataset = fn_get_training_dataset()

    #------------------------------------------------------------------
    # Cleaning Data  + Remove Bubble games in id_season=2020
    training_dataset = CleaningData(training_dataset)

    #------------------------------------------------------------------
    # Features Engineering
    training_dataset = FeaturesEngineeringPipeline(training_dataset)

    #------------------------------------------------------------------
    # Y variables function
    training_dataset = fn_get_y_variables(training_dataset)

    #------------------------------------------------------------------
    # Final output 1: Predict proba team to win (2 lines per games)
    # FinalTest
    GetTrainingDataset2LinePerGames(training_dataset)

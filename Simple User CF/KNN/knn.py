from helper import *
from surprise import KNNBasic, KNNWithMeans, KNNBaseline, accuracy
from surprise import NormalPredictor
from surprise.model_selection import train_test_split, GridSearchCV, KFold
import random
import pandas as pd
import numpy as np
np.random.seed(0)
random.seed(0)

data = GetBookData(density_filter = True)
trainset, testset = train_test_split(data, test_size=0.2)
results = {}

def KNN_Tester(trainset,testset,algo):
    param_grid = {'k': [50, 100],
                  'sim_options': {'name': ['msd', 'cosine','pearson']}
                  }
                  
    gs = GridSearchCV(algo, param_grid, measures = ['rmse'], cv = 5)
    gs.fit(data)
    params = gs.best_params['rmse']
    algo = KNNBasic(k = params['k'], 
                    sim_options = params['sim_options'])
    algo.fit(trainset)
    predictions  = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    precisions, recalls = precision_recall_at_k(pred, k = 10, threshold = 4)
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
    metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
    return metrics

results['KNN_Basic'] = KNN_Tester(trainset,testset, KNNBasic)
results['KNN_Baseline'] = KNN_Tester(trainset,testset, KNNBasic)
results['KNN_With_Means'] = KNN_Tester(trainset,testset,KNNWithMeans)


#
###KNNBasic - Cosine
#sim_options = {'name': 'cosine',
#               'user_based': True}
#algo = KNNBasic(sim_options=sim_options)
#algo.fit(trainset)
#pred = algo.test(testset)
#precisions, recalls = precision_recall_at_k(pred, k = 10, threshold = 4)
#avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
#avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
#results['KNNBasicCosine'] = {'RMSE':accuracy.rmse(pred), 
#       'Avg_Precision': avg_precision, 
#       'Avg_Recall': avg_recall}
#
###KNNBasic - MSD
#sim_options = {'name': 'msd',
#               'user_based': True}
#algo = KNNBasic(sim_options=sim_options)
#algo.fit(trainset)
#pred = algo.test(testset)
#precisions, recalls = precision_recall_at_k(pred, k = 10, threshold = 4)
#avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
#avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
#results['KNNBasicMSD'] = {'RMSE':accuracy.rmse(pred), 
#       'Avg_Precision': avg_precision, 
#       'Avg_Recall': avg_recall}
#
###KNNBasic - MSD
#sim_options = {'name': 'msd',
#               'user_based': True}
#algo = KNNBasic(sim_options=sim_options)
#algo.fit(trainset)
#pred = algo.test(testset)
#precisions, recalls = precision_recall_at_k(pred, k = 10, threshold = 4)
#avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
#avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
#results['KNNBasicMSD'] = {'RMSE':accuracy.rmse(pred), 
#       'Avg_Precision': avg_precision, 
#       'Avg_Recall': avg_recall}
#
#
###KNNBasic - Pearson
#sim_options = {'name': 'Pearson',
#               'user_based': True}
#algo = KNNBasic(sim_options=sim_options)
#algo.fit(trainset)
#pred = algo.test(testset)
#precisions, recalls = precision_recall_at_k(pred, k = 10, threshold = 4)
#avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
#avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
#results['KNNBasicPearson'] = {'RMSE':accuracy.rmse(pred), 
#       'Avg_Precision': avg_precision, 
#       'Avg_Recall': avg_recall}    

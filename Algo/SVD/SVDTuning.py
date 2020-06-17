# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from helper import *
from surprise import SVD, NormalPredictor, accuracy
from surprise import NormalPredictor
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import random
import numpy as np
np.random.seed(0)
random.seed(0)


data = GetBookData(density_filter = False)
trainset, testset = train_test_split(data, test_size=0.25)


##Tuning Parameters
param_grid = {'n_epochs': [30, 30], 
            'lr_all': [0.001, 0.15],
              'reg_all':[0.01,0.1],
              'n_factors': [10, 200]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
gs.fit(data)
params = gs.best_params['rmse']
SVD_TUNED = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
SVD_TUNED.fit(trainset)
gs_predictions = SVD_TUNED.test(testset)
rmse = accuracy.rmse(gs_predictions)

precisions, recalls = precision_recall_at_k(gs_predictions, k = 10, threshold = 4.9)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)


metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
results['SVD'] = metrics


#
#print("Searching for best parameters...")
#param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
#              'n_factors': [50, 100]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
#
#gs.fit(data)
#
## best RMSE score
#print("Best RMSE score attained: ", gs.best_score['rmse'])
#
## combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])
#
#
#params = gs.best_params['rmse']
#SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
#
#SVDUntuned = SVD()
#
## Just make random recommendations
#Random = NormalPredictor()




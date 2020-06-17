from helper import *
from surprise import NormalPredictor, BaselineOnly, accuracy
from surprise.model_selection import train_test_split, GridSearchCV, KFold
import random
import pandas as pd
import numpy as np
np.random.seed(0)
random.seed(0)
pd.set_option('display.max_columns', 500)

data = GetBookData(density_filter = False)
trainset, testset = train_test_split(data, test_size=0.2)
results = {}
top_n = {}

norm = NormalPredictor()
norm.fit(trainset)
norm_pred = norm.test(testset)
rmse = accuracy.rmse(norm_pred)
precisions, recalls = precision_recall_at_k(norm_pred, k = 10, threshold = 4)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
metrics = {'rmse': rmse, 
            'avg_precision': avg_precision, 
            'avg_recall': avg_recall}
results['NormalPredictor'] = metrics

top_n['NormalPredictor'] = get_top_n(norm_pred, n=10)


param_grid = {'bsl_options':{'method': ['als', 'sgd']}}
gs = GridSearchCV(BaselineOnly, param_grid, measures = ['rmse'], cv = 5)
gs.fit(data)
params = gs.best_params['rmse']
algo = BaselineOnly(bsl_options = params['bsl_options'])
algo.fit(trainset)
base_pred  = algo.test(testset)
rmse = accuracy.rmse(base_pred)
precisions, recalls = precision_recall_at_k(base_pred, k = 10, threshold = 4)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
results['BaselineOnly'] = metrics

top_n['BaselineOnly'] = get_top_n(base_pred, n=10)



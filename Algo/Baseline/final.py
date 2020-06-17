from helper import *
from surprise import NormalPredictor, BaselineOnly, accuracy, KNNBasic, KNNWithMeans, KNNBaseline, SVD
from surprise.model_selection import train_test_split, GridSearchCV, KFold
import random
import pandas as pd
import numpy as np
np.random.seed(0)
random.seed(0)
pd.set_option('display.max_columns', 500)

data,items,ratings = GetBookData(density_filter = True)
trainset, testset = train_test_split(data, test_size=0.2)
results = {}
top_n = {}


###Normal Predictor
norm = NormalPredictor()
norm.fit(trainset)
norm_pred = norm.test(testset)
rmse = accuracy.rmse(norm_pred)
precisions, recalls = precision_recall_at_k(norm_pred, k = 10, threshold = 4.5)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
metrics = {'rmse': rmse, 
            'avg_precision': avg_precision, 
            'avg_recall': avg_recall}
results['NormalPredictor'] = metrics

top_n['NormalPredictor'] = get_top_n(norm_pred, n=10)

###Baseline Predictor
param_grid = {'bsl_options':{'method': ['als', 'sgd']}}
gs = GridSearchCV(BaselineOnly, param_grid, measures = ['rmse'], cv = 5)
gs.fit(data)
params = gs.best_params['rmse']
algo = BaselineOnly(bsl_options = params['bsl_options'])
algo.fit(trainset)
base_pred  = algo.test(testset)
rmse = accuracy.rmse(base_pred)
precisions, recalls = precision_recall_at_k(base_pred, k = 10, threshold = 4.5)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
results['BaselineOnly'] = metrics

top_n['BaselineOnly'] = get_top_n(base_pred, n=10)

###KNN

def KNN_Tester(trainset,testset,algo):
    param_grid = {'k': [1, 50],
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
    precisions, recalls = precision_recall_at_k(predictions, k = 10, threshold = 4.5)
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
    metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
    topn = get_top_n(predictions, n = 10)
    return metrics,topn

results['KNN_Basic'], top_n['KNN_Basic'] = KNN_Tester(trainset,testset, KNNBasic)
results['KNN_Baseline'], top_n['KNN_Baseline'] = KNN_Tester(trainset,testset, KNNBasic)
results['KNN_With_Means'], top_n['KNN_With_Means'] = KNN_Tester(trainset,testset,KNNWithMeans)

###SVD
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
precisions, recalls = precision_recall_at_k(gs_predictions, k = 10, threshold = 4.5)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)


metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall,
               'best_parameters': params}
results['SVD'] = metrics
top_n['SVD'] = get_top_n(gs_predictions, n=10)


###SVD Out of Box
clf = SVD()
clf.fit(trainset)
svd_pred = clf.test(testset)
rmse = accuracy.rmse(svd_pred)
precisions, recalls = precision_recall_at_k(svd_pred, k = 10, threshold = 4.5)
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
metrics = {'rmse': rmse, 
               'avg_precision': avg_precision, 
               'avg_recall': avg_recall}
results['SVD_OOB'] = metrics

top_n['SVD_OOB'] = get_top_n(base_pred, n=10)



unique_users = ratings['userID'].unique()


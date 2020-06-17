# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:03:41 2019

@author: Jon
"""


from helper import *
from surprise import SVD, NormalPredictor, accuracy
from surprise import NormalPredictor
from surprise.model_selection import train_test_split, GridSearchCV, KFold
import random
import numpy as np
np.random.seed(0)
random.seed(0)

GetBookData(density_filter = False)
data = GetBookData(density_filter = False)
trainset, testset = train_test_split(data, test_size=0.25)

##SVD Out of the Box
SVD_OOB = SVD()
SVD_OOB.fit(trainset)
oob_predictions = SVD_OOB.test(testset)
oob_rmse = accuracy.rmse(oob_predictions)
oob_mae = accuracy.mae(oob_predictions)

precisions, recalls = precision_recall_at_k(oob_predictions, k = 10, threshold = 4)
oob_avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
oob_avg_recall= sum(rec for rec in recalls.values()) / len(recalls)
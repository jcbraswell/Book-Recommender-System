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




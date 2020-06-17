# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:18:05 2019

@author: Jon
"""
from collections import defaultdict
import pandas as pd
from surprise import Reader, Dataset
#import numpy as np

def GetBookData(density_filter):
    items = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    items.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    users = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ['userID', 'Location', 'Age']
    users['userID'] = users['userID'].apply(lambda x: str(x))
    ratings = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']
    ratings['userID'] = ratings['userID'].apply(lambda x: str(x))
    #Getting rid of implicit ratings
    ratings = ratings[ratings['bookRating'] > 0]
    ratings = ratings[ratings['ISBN'].isin(items['ISBN'])]
    
    if density_filter:
        #To reduce our dataset we are going to remove items which were rated less than 10 times
        a = ratings.groupby('ISBN').filter(lambda x: len(x) >= 10) 
#        Remove users who gave less than 20 ratings
        ratings = a.groupby('userID').filter(lambda x: len(x) >= 20)
    else:
        reader = Reader(rating_scale=(1,10))
        data = Dataset.load_from_df(ratings,reader)
    return data


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls



def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

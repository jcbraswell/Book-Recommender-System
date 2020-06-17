# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:18:05 2019

@author: Jon
"""
import pandas as pd
from surprise import Reader, Dataset
#import numpy as np

def GetBookData(items,ratings,users):
#     items = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
#     items.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
#     users = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
#     users.columns = ['userID', 'Location', 'Age']
#     users['userID'] = users['userID'].apply(lambda x: str(x))
#     ratings = pd.read_csv('C:/Users/Jon/Documents/Statistics/Machine Learning/Project/Book Crossing/BX Project/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
#     ratings.columns = ['userID', 'ISBN', 'bookRating']
#     ratings['userID'] = ratings['userID'].apply(lambda x: str(x))
    
    
    ##Checking size of data
    #print('Number of Users: ',users.shape[0])
    #print('Number of Items: ',items.shape[0])
    #print('Number of Ratings: ',ratings.shape[0])
    #print('Number of Users in Ratings: ', len(ratings['userID'].unique()))
    #print('Number of Items in Ratings: ', len(ratings['ISBN'].unique()))
    
    
    #Getting rid of implicit ratings
    ratings = ratings[ratings['bookRating'] > 0]
    
    ratings = ratings[ratings['ISBN'].isin(items['ISBN'])]
#    
#    density = (float(len(ratings))/(len(np.unique(ratings['userID']))*len(np.unique(ratings['ISBN']))))*100
#    #print("Density in percent: "+str(density) )
    #print("Users: "+str(len(np.unique(ratings['userID'])))+ " items: "+str(len(np.unique(ratings['ISBN']))))
    
    #To reduce our dataset we are going to remove items which were rated less than 10 times
    a = ratings.groupby('ISBN').filter(lambda x: len(x) >= 10)
#    densityi = (float(len(a))/(len(np.unique(a['userID']))*len(np.unique(a['ISBN']))))*100
    #print("Density after filtering items: "+str(densityi))
    #print("Users: "+str(len(np.unique(a['userID'])))+ " items: "+str(len(np.unique(a['ISBN']))))
#     reader = Reader(rating_scale=(1,10))
#     data = Dataset.load_from_df(a,reader)    
    
    #Remove users who gave less than 20 ratings
    ratings = a.groupby('userID').filter(lambda x: len(x) >= 20)
#    densityu = (float(len(ratings))/(len(np.unique(ratings['userID']))*len(np.unique(ratings['ISBN']))))*100
    #print("Density after filtering users: "+str(densityu))
    #print("Users: "+str(len(np.unique(ratings['userID'])))+ " items: "+str(len(np.unique(ratings['ISBN']))))
    reader = Reader(rating_scale=(1,10))
    data = Dataset.load_from_df(ratings,reader)
    
    
    return data


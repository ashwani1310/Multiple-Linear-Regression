#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 05:36:04 2017

@author: ashwani
"""

#The code is based upon the foolowing assumption that the last column of features contain 
#a categorical data with three categories.


#To Import the needed libraries which will help us build our model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#
#Here, the data set is being imported and the features and labels 
#are stored in separate arrays
data = pd.read_csv('any csv file to be added here containing multiple input feature')
Features = data.iloc[:, :-1].values # here the label is the last column and the rest are features.
Label = data.iloc[:, 4].values

#Now, to encode the categorical data into dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
feature_encoder = LabelEncoder()
Features[:,3] = feature_encoder.fit_transform(Features[:,3])
feature_one_encoder = OneHotEncoder(categorical_features = [3])
Features = feature_one_encoder.fit_transform(Features).toarray()                 

#Always remove one dummy variable from total       
Features = Features[:, 1:]

#This is to split training and testing data from the imported dataset          
from sklearn.cross_validation import train_test_split
Features_train, Features_test, Label_train, Label_test = train_test_split(Features, Label, test_size = 0.2, random_state = 0)

#Now making the model
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(Features_train,Label_train)

#Now to test the model results
Label_pred = reg_model.predict(Features_test)

#Optimizing our model using backward elimination
import statsmodels.formula.api as stat
Features = np.append(arr = np.ones((50,1)).astype(int), values = Features, axis = 1)
Features_optimal = Features[:, [0,1,2,3,4,5]]
reg_model_least_squares = stat.OLS(endog = Label, exog = Features_optimal).fit()
reg_model_least_squares.summary()
Features_optimal = Features[:, [0, 1, 3, 4, 5]]
reg_model_least_squares = stat.OLS(endog = Label, exog = Features_optimal).fit()
reg_model_least_squares.summary()
Features_optimal = Features[:, [0, 3, 4, 5]]
reg_model_least_squares = stat.OLS(endog = Label, exog = Features_optimal).fit()
reg_model_least_squares.summary()
Features_optimal = Features[:, [0, 3, 5]]
reg_model_least_squares = stat.OLS(endog = Label, exog = Features_optimal).fit()
reg_model_least_squares.summary()
Features_optimal = Features[:, [0, 3]]
reg_model_least_squares = stat.OLS(endog = Label, exog = Features_optimal).fit()
reg_model_least_squares.summary()
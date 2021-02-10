"""
Final Project: CE 5319
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 12 / 09 / 2020

# The model uses cross validation for deep neural network
# It uses concrete_clean.csv
# Total Raw Data initial = 1030
# Total Data after cleaning = 968

"""
# Step 1: Importing the required packages

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt
import hydroeval as hyd
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

# Checking the Working Directory

"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""

# Reading the dataset

a = pd.read_csv('concrete_clean.csv') 
features = ['AgeDays', 'W.C', 'Cement', 'Water','FineAgg','CoarseAgg',
            'BlastFurnaceSlag','FlyAsh','Superplasticizer']
X = a[features]
Y = a['StrengthMPa']

# Step 4: Applying Data Scaling
scaler = StandardScaler() # Initializing standardizer

X_scl = scaler.fit_transform(X)
Y_scl = scale(Y)

# Step 5: Splitting into 75% training and 25% testing set

X_train, X_test, Y_train, Y_test = train_test_split(X_scl,Y_scl, test_size = 0.25, random_state = 10)

# Setting up the tensorflow model in keras

def create_model():
    ker = Sequential()
    ker.add(Dense(27, input_dim = 9, activation='relu'))
    ker.add(Dense(9, activation='relu'))
    ker.add(Dense(1))
    ker.compile(loss = 'mean_squared_error', metrics=['mae'], optimizer = 'adam')
    return ker

# Step 8: Fitting the model
ker = KerasRegressor(build_fn=create_model, epochs=250, batch_size=15, verbose=0)
scores = cross_val_score(ker, X_train, Y_train, scoring='neg_mean_squared_error', cv=5)
mean_cv = scores.mean()

# Printing the cross - validation scores

print("The scores of cross-validation: ", scores)
print("Mean cross validation value: ", mean_cv)
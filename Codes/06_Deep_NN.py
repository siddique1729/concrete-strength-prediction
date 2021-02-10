"""
Final Project: CE 5319
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 12 / 09 / 2020

# The model uses deep neural network
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
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

# Step 2: Checking the Working Directory

"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""

# Step 3: Reading the dataset

# Reading the Dataset
# Using the dataset 'concrete_clean' after cleaning

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

# Step 6: Setting up the tensorflow model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(27, input_dim = 9, activation = 'relu'), #Hidden layer 1
    tf.keras.layers.Dense(9, activation = 'relu', #Hidden layer 2
    kernel_regularizer = tf.keras.regularizers.L2(0.01)), # Regularizer addition
    tf.keras.layers.Dense(1),]) # Output layer

# Step 7: Model Compilation

model.compile(loss = 'mean_squared_error', 
              metrics=['mae'], 
              optimizer = 'adam')

# Step 8: Fitting the model

history = model.fit(X_train, Y_train, epochs = 250, verbose = 1, batch_size = 15)

# Plotting the model accuracy and error

plt.figure(0) # numbering the plot - signifying the beginning of current plot
line1 = plt.plot(history.history['loss'], label = 'Loss'),
line2 = plt.plot(history.history['mae'], label = 'mae'),
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend(loc="best")
plt.grid()
plt.title('Training Performance')

# Evaluating training Fits

plt.figure(1)
y_tp = model.predict(X_train)
plt.scatter(Y_train, y_tp)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Training Evaluation')
plt.grid()


# Making predictions and test predictions

plt.figure(2)
y_pred = model.predict(X_test)
plt.scatter(Y_test, y_pred)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Testing Evaluation')
plt.grid()


# Computing advanced metrics 
# Uses Kling Gupta Evaluation Metrics

y_tp = y_tp.flatten()
y_pred = y_pred.flatten()

trainy = np.array(Y_train) # conversion to numpy arrays
testy = np.array(Y_test) # conversion to numpy arrays

kge_train = hyd.kgeprime(y_tp,trainy) # computing metric for train data
kge_test = hyd.kgeprime(y_pred,testy) # computing metric for test data

# User defined function to get the basic metrics of the model 

def metricx(observed,predicted):
    mse = metrics.mean_squared_error(observed,predicted)
    mae = metrics.mean_absolute_error(observed,predicted)
    cor = np.corrcoef(observed,predicted)
    
    zz = [mse, mae, cor]
    return(zz)

# Calling the UDF for metrics calculation

train_metric = metricx(Y_train, y_tp)
test_metric = metricx(Y_test, y_pred)

# printing the metric values

print("The Kling Gupta Metric for Train Data: ", kge_train)
print("The Kling Gupta Metric for Test Data: ", kge_test)
print("The Summary of Metric Accuracy for Train Data: ",train_metric)
print("The Summary of Metric Accuracy for Test Data: ", test_metric)
"""
Final Project: CE 5319
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 12 / 09 / 2020

# The model uses linear regression
# It uses concrete_clean.csv
# Total Raw Data initial = 1030
# Total Data after cleaning = 968

"""

# importing all libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import hydroeval as hyd

# Reading the Dataset

a = pd.read_csv('concrete_clean.csv')
X = a.iloc[:,0:9] # Dataframe of the 9 input features
Y = a['StrengthMPa'] # Series of the output feature - StrengthMPa

# Step 4: Applying Data Scaling
scaler = StandardScaler() # Initializing standardizer

X_scl = scaler.fit_transform(X)
Y_scl = scale(Y)


# Splitting the dataset into 75% training and 25% testing data

X_train, X_test, Y_train, Y_test = train_test_split(X_scl, Y_scl, test_size = 0.25, random_state = 10)

# fitting the linear regression model
# running RFE with n number of features

lm = LinearRegression()
lm.fit(X_train, Y_train)

rfe = RFE(lm, n_features_to_select = 9)             
rfe = rfe.fit(X_train, Y_train)

y_pred = rfe.predict(X_test)

# Cross Validation

scores = cross_val_score(lm, X_train, Y_train, scoring='neg_mean_squared_error', cv=5)
mean_cv = scores.mean()

# Plotting the model accuracy and error

# Evaluating training Fits

plt.figure(1)
y_tp = rfe.predict(X_train)
plt.scatter(Y_train, y_tp)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Training Evaluation')
plt.grid()


# Making predictions and test predictions

plt.figure(2)
y_pred = rfe.predict(X_test)
plt.scatter(Y_test, y_pred)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Testing Evaluation')
plt.grid()

# Computing advanced metrics 
# USes Kling Gupta Evaluation Metrics

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
print("The scores of cross-validation: ", scores)
print("Mean cross validation value: ", mean_cv)
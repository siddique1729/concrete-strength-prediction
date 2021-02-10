"""
Final Project: CE: 5319
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 12 / 09 / 2020

# The model uses Gradient Boosting Regression Model
# It uses concrete_clean.csv
# Total Raw Data initial = 1030
# Total Data after cleaning = 968
"""
# Loading Libraries
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
import seaborn as sns
from matplotlib import pyplot as plt
import hydroeval as hyd
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# Checking Directory
"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""

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

# Performing Grid Search for Best Hyperparameters using 5-fold CV

gsc = GridSearchCV(
    estimator=AdaBoostRegressor(),param_grid={'learning_rate': np.arange(0.1,1),
   'n_estimators': (10, 50, 100, 200, 500)},cv=5, n_jobs=-1)
grid_result = gsc.fit(X_train, Y_train)
best_params = grid_result.best_params_

# Fitting the Ada Boost model with train data and predict testing

rfr= AdaBoostRegressor(n_estimators=best_params['n_estimators'], 
                         learning_rate=best_params['learning_rate']) 
rfr.fit(X_train,Y_train)
y_pred = rfr.predict(X_test)
train_pred = rfr.predict(X_train)

# Feature Importance
names = list(X.columns)# Get names of variables
imp = rfr.feature_importances_ # Obtain feature importance
impa = (names,imp) # Make a tuple
impadf = pd.DataFrame(impa) # Write to a dataframe

# Relative Importance Plot
plt.figure(0)
sns.set(style="whitegrid")
ax = sns.barplot(x=imp, y=names)
ax.set(xlabel="Relative Importance")

# Computing advanced metrics 
# USes Kling Gupta Evaluation Metrics

trainy = np.array(Y_train) # conversion to numpy arrays
testy = np.array(Y_test) # conversion to numpy arrays

kge_train = hyd.kgeprime(train_pred,trainy) # computing metric for train data
kge_test = hyd.kgeprime(y_pred,testy) # computing metric for test data

# User defined function to get the basic metrics of the model 

def metricx(observed,predicted):
    mse = metrics.mean_squared_error(observed,predicted)
    mae = metrics.mean_absolute_error(observed,predicted)
    cor = np.corrcoef(observed,predicted)
    
    zz = [mse, mae, cor]
    return(zz)

# Calling the UDF for metrics calculation

train_metric = metricx(Y_train, train_pred)
test_metric = metricx(Y_test, y_pred)

# Evaluating training Fits

plt.figure(1)
plt.scatter(Y_train, train_pred)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Training Evaluation')
plt.grid()


# Evaluating test predictions

plt.figure(2)
plt.scatter(Y_test, y_pred)
plt.plot(Y_test, Y_test, c ='red')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Testing Evaluation')
plt.grid()

# printing the metric values

print("The Kling Gupta Metric for Train Data: ", kge_train)
print("The Kling Gupta Metric for Test Data: ", kge_test)
print("The Summary of Metric Accuracy for Train Data: ",train_metric)
print("The Summary of Metric Accuracy for Test Data: ", test_metric)
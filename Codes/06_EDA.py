"""
Final Project: CE: 5319
Author: The code is written by Jawwad Shadman Siddique | R11684947
Date of Submission: 12 / 09 / 2020

# The raw dataset is named as 'concrete.csv'
# The dataset is cleaned in R be removing outliers
# The used dataset for the exploratory data analysis is concrete_clean.csv
# Total raw data = 1030
# Total Data after cleaning = 968

# Lifecycle in a Data Science Project
1.Data Analysis 
2.Feature Engineering 
3.Feature Selection 
4.Model Building 
5.Model Deployment

# In Data Analysis step we will analyze to find out the below metrics

1. Missing Values 2. Data Type in each column 3. Descriptive Statistics Summary 
4. Variable Types Count for Numerical Variable 5. Boxplot of the entire dataset 
6. Boxplot of Dependent Variable vs Discrete Independent Variable 
7. Histogram Distribution of Dependent Variable vs Discrete Independent Variable 
8. Histogram Distribution plots of all independent continuous variables 
9. Swarmplot & Violinplot for the dependent variable 10. Skewness and Kurtosis 
11. Pair Scatter Plot 12. Joint Plot (Kde distribution included) 
13. ECDF 14. Correlation Matrix 15. Model Entropy

"""
# Loading the libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

# Checking Directory
"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""
# Reading the Dataset

dataset = pd.read_csv('concrete_clean.csv')

# Dividing the dataset into two parts for doing the joint and pair plot

x1 = dataset[['Cement','Water','CoarseAgg', 'FineAgg', 'W.C','StrengthMPa']].copy()
x2 = dataset[['BlastFurnaceSlag','FlyAsh','Superplasticizer','StrengthMPa']].copy()

# print shape of dataset with rows and columns
print(dataset.shape)

# print the top 5 records
dataset.head()

# checking if there are missing values
miss_value = dataset.isnull().sum()
print(miss_value)

# checking the types of data (numerical or not)
dataset.info()

# checking the columns
dataset.columns

#descriptive statistics summary
descr = dataset.describe()
print(descr)

# Numerical variables are of two types
# 1. Continuous Variable 2. Discrete Variable

# list of numerical variables

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print(numerical_features)
print('Number of numerical variables: ', len(numerical_features))

# list of discrete variables
discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
print(discrete_feature)

# Continuous Variable
# continuous_feature=[feature for feature in numerical_features 
# if feature not in discrete_feature+['StrengthMPa']]
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))

# Boxplots of all variables of the entire dataset
plt.figure(0)
ax = sns.boxplot(data = dataset, orient="h", palette="Set1")

# Box plot of Strength against Age
var = 'AgeDays'
data = pd.concat([dataset['StrengthMPa'], dataset[var]], axis=1)
plt.figure(1)
fig = sns.boxplot(x=var, y="StrengthMPa", data=dataset)

# Distribution of Strength with respect to Age of Testing
plt.figure(2)
sns.set_theme(style="whitegrid")
sns.displot(dataset, x = 'StrengthMPa', hue="AgeDays", stat = "density", palette = 'bright')


# Distribution and normal probability plot with skewness and Kurtosis

# Strength 
plt.figure(3) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['StrengthMPa'], fit=norm)

plt.figure(4) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['StrengthMPa'], plot=plt)

print("Skewness: %f" % dataset['StrengthMPa'].skew()) # Skewness
print("Kurtosis: %f" % dataset['StrengthMPa'].kurt()) # Kurtosis

# Cement 
plt.figure(5) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['Cement'], fit=norm)

plt.figure(6) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['Cement'], plot=plt)

print("Skewness: %f" % dataset['Cement'].skew()) # Skewness
print("Kurtosis: %f" % dataset['Cement'].kurt()) # Kurtosis

# BlastFurnaceSlag 
plt.figure(7) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['BlastFurnaceSlag'], fit=norm)

plt.figure(8) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['BlastFurnaceSlag'], plot=plt)

print("Skewness: %f" % dataset['BlastFurnaceSlag'].skew()) # Skewness
print("Kurtosis: %f" % dataset['BlastFurnaceSlag'].kurt()) # Kurtosis

# FlyAsh
plt.figure(9) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['FlyAsh'], fit=norm)

plt.figure(10) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['FlyAsh'], plot=plt)

print("Skewness: %f" % dataset['FlyAsh'].skew()) # Skewness
print("Kurtosis: %f" % dataset['FlyAsh'].kurt()) # Kurtosis

# Water 
plt.figure(11) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['Water'], fit=norm)

plt.figure(12) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['Water'], plot=plt)

print("Skewness: %f" % dataset['Water'].skew()) # Skewness
print("Kurtosis: %f" % dataset['Water'].kurt()) # Kurtosis

# Superplasticizer 
plt.figure(13) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['Superplasticizer'], fit=norm)

plt.figure(14) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['Superplasticizer'], plot=plt)

print("Skewness: %f" % dataset['Superplasticizer'].skew()) # Skewness
print("Kurtosis: %f" % dataset['Superplasticizer'].kurt()) # Kurtosis

# Coarse Aggregate
plt.figure(15) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['CoarseAgg'], fit=norm)

plt.figure(16) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['CoarseAgg'], plot=plt)

print("Skewness: %f" % dataset['CoarseAgg'].skew()) # Skewness
print("Kurtosis: %f" % dataset['CoarseAgg'].kurt()) # Kurtosis

# Fine Aggregate
plt.figure(17) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['FineAgg'], fit=norm)

plt.figure(18) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['FineAgg'], plot=plt)

print("Skewness: %f" % dataset['FineAgg'].skew()) # Skewness
print("Kurtosis: %f" % dataset['FineAgg'].kurt()) # Kurtosis

# Water Cement Ratio 
plt.figure(19) # Distribution
sns.set_theme(style="whitegrid")
sns.distplot(dataset['W.C'], fit=norm)

plt.figure(20) # Probability plot
sns.set_theme(style="whitegrid")
res = stats.probplot(dataset['W.C'], plot=plt)

print("Skewness: %f" % dataset['W.C'].skew()) # Skewness
print("Kurtosis: %f" % dataset['W.C'].kurt()) # Kurtosis


# Swarmplot for StrengthMPa 

plt.figure(21)
sns.set_theme(style="whitegrid")
sns.swarmplot(y = dataset['StrengthMPa'], orient=('v'), palette = "Set2")

# Swarmplots on Violinplots for Strength per Agedays

plt.figure(22)
sns.set_theme(style="whitegrid")
ax = sns.violinplot(x="AgeDays", y="StrengthMPa", data=dataset, inner=None)
ax = sns.swarmplot(x="AgeDays", y="StrengthMPa", data=dataset,
                   palette="Set2", edgecolor="gray")


#scatterplot1 for - 'Cement', 'Water','CoarseAgg', 'FineAgg', 'W/C', 'StrengthMPa'
plt.figure(23)
sns.set_theme(style="whitegrid")
sns.pairplot(x1, size = 2.5)
plt.show()

#scatterplot2 for - 'BlastFurnaceSlag', 'FlyAsh', 'Superplasticizer','StrengthMPa'
plt.figure(24)
sns.set_theme(style="whitegrid")
sns.pairplot(x2, size = 2.5, palette = 'Set1' )
plt.show()

# Joint Plot 1 - 'Cement', 'Water','CoarseAgg', 'FineAgg', 'W/C', 'StrengthMPa'
# Kde distributions included

plt.figure(25)
sns.set_theme(style="whitegrid")
g = sns.PairGrid(x1)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)

# Joint Plot 2 - 'BlastFurnaceSlag', 'FlyAsh', 'Superplasticizer','StrengthMPa'
# Kde distributions included

plt.figure(26)
sns.set_theme(style="whitegrid")
g = sns.PairGrid(x2)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)

# Empirical Cumulative Distribution Function

plt.figure(27)
sns.set_theme(style="whitegrid")
sns.displot(dataset, x="StrengthMPa", kind="ecdf")

# Correlation matrix

plt.figure(28)
corrmat = dataset.corr()
sns.heatmap(corrmat, vmax=.8, square=True)

# Strength correlation matrix

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'StrengthMPa')['StrengthMPa'].index
cm = np.corrcoef(dataset[cols].values.T)
plt.figure(29)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Calculating Model Entropy

X = dataset.iloc[:,0:9] # Dataframe of the 9 input features
Y = dataset['StrengthMPa'] # Series of the output feature - StrengthMPa

MI = mutual_info_regression(X,Y)
MI = MI*100/np.max(MI)


# Plot Showing important variables based on Mutual Information

cols = list(dataset.columns)[0:9]
fig = plt.figure(23)
ax = fig.add_axes([0,0,1,1])
ax.bar(cols,MI)
plt.ylabel('Rel. Mutual Information')
plt.xticks(rotation='vertical')
plt.grid(True)
plt.show()
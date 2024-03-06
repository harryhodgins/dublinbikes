import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

#IMPORT DATA
data = pd.read_csv("train_data_merged_noholiday.csv")
#data = pd.read_csv("train_data_merged.csv")


# Selecting features and target variable
X = data.drop(['DATE', 'AVERAGE_DAILY_USAGE'], axis=1)
Y = data['AVERAGE_DAILY_USAGE']

#KFOLD CROSS VALIDATION
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor

kf = KFold(n_splits = 5,shuffle = True)

C_values = [0.001,0.01,1,5,500,1000] #penalty parameters
mean_error = []
std_error = []
mean_error_r2 = []
std_error_r2 = []
mean_error_ridge = []
std_error_ridge = []
mean_error_r2_ridge = []
std_error_r2_ridge = []
mean_error_dummy = []
std_dummy = []
mean_error_r2_dummy = []
std_r2_dummy = []

for c in C_values:
    
    dummy_reg = DummyRegressor(strategy = 'mean')
    model_lasso = Lasso(alpha = 1/(2*c),max_iter=10000)
    model_ridge = Ridge(alpha = 1/(2*c),max_iter = 10000)
    
    error = []
    error_ridge = []
    r2_error = []
    r2_error_ridge = []
    error_dummy = []
    r2_error_dummy = []
    error_ridge_dummy = []
    
    for train, test in kf.split(X.values):
        
        model_lasso.fit(X.iloc[train].values,Y[train])
        model_ridge.fit(X.iloc[train].values,Y[train])
        dummy_reg.fit(X.iloc[train].values,Y[train])
        
        pred_lasso = model_lasso.predict(X.iloc[test].values)
        pred_ridge = model_ridge.predict(X.iloc[test].values)
        pred_dummy = dummy_reg.predict(X.iloc[test].values)
        
        error.append(median_absolute_error(Y[test],pred_lasso))
        r2_error.append(r2_score(Y[test],pred_lasso))
        
        error_ridge.append(median_absolute_error(Y[test],pred_ridge))
        r2_error_ridge.append(r2_score(Y[test],pred_ridge))
        
        error_dummy.append(median_absolute_error(Y[test],pred_dummy))
        r2_error_dummy.append(r2_score(Y[test],pred_dummy))
        
    mean_error_r2.append(np.array(r2_error).mean()) #mean error
    std_error_r2.append(np.array(r2_error).std()) #standard deviation in error
    mean_error.append(np.array(error).mean()) #mean error
    std_error.append(np.array(error).std())
    
    mean_error_r2_ridge.append(np.array(r2_error_ridge).mean()) #mean error
    std_error_r2_ridge.append(np.array(r2_error_ridge).std()) #standard deviation in error
    mean_error_ridge.append(np.array(error_ridge).mean()) #mean error
    std_error_ridge.append(np.array(error_ridge).std())
    
    mean_error_r2_dummy.append(np.array(r2_error_dummy).mean()) #mean error
    std_r2_dummy.append(np.array(r2_error_dummy).std()) #standard deviation in error
    mean_error_dummy.append(np.array(error_dummy).mean()) #mean error
    std_dummy.append(np.array(error_dummy).std())

print("------LASSO------")    
print("Max r2 score: %s"%max(mean_error_r2))
print("Min error: %s"%min(mean_error))

print("------RIDGE------")    
print("Max r2 score: %s"%max(mean_error_r2_ridge))
print("Min error: %s"%min(mean_error_ridge))

print("------DUMMY------")    
print("Max r2 score: %s"%max(mean_error_r2_dummy))
print("Min error: %s"%min(mean_error_dummy))

#plotting mean/std vs. C
plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error,yerr= std_error,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('MAD',fontsize=16)
plt.title("Median Absolute Deviation vs. C - LASSO",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()

plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error_r2,yerr= std_error_r2,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('R2 Score',fontsize=16)
plt.title("R2 Score vs. C - LASSO",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()

plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error_ridge,yerr= std_error_ridge,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('MAD',fontsize=16)
plt.title("Median Absolute Deviation vs. C - Ridge Regression",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()

plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error_r2_ridge,yerr= std_error_r2_ridge,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('R2 Score',fontsize=16)
plt.title("R2 Score vs. C - Ridge Regression",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()

plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error_dummy,yerr= std_dummy,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('MAD',fontsize=16)
plt.title("Median Absolute Deviation vs. C - Baseline Regressor",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()

plt.figure(figsize = (10,5))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.errorbar(C_values,mean_error_r2_dummy,yerr= std_r2_dummy,ecolor = 'red')
plt.xlabel('C',fontsize=16)
plt.ylabel('R2 Score',fontsize=16)
plt.title("R2 Score vs. C - Baseline Regressor",fontsize=16)
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.show()


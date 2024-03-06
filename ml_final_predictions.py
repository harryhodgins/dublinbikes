import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#IMPORT DATA
#data = pd.read_csv("mlhw_final/data_merged.csv")
data = pd.read_csv("data_bikes_holidays.csv")


data['DATE'] = pd.to_datetime(data['DATE'])



#dataframe for visualising the actual usage
totals_df = pd.DataFrame({'Date':data['DATE'],'usage':data['AVERAGE_DAILY_USAGE']})

plt.figure(figsize=(15, 7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(totals_df['Date'], totals_df['usage'], label='Actual Usage')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Bike Usage',fontsize=18)
plt.title('Bike Usage - Pre-Pandemic & Pandemic Period',fontsize=20)
plt.legend(fontsize=12)
plt.show()


#train data will be up to march 1st
date_threshold = pd.to_datetime('2020-03-01')
train_data = data[data['DATE'] < date_threshold]
test_data = data[data['DATE'] >= date_threshold]

X_train = train_data.drop(['DATE', 'AVERAGE_DAILY_USAGE'], axis=1).values
Y_train = train_data['AVERAGE_DAILY_USAGE']

X_test = test_data.drop(['DATE', 'AVERAGE_DAILY_USAGE'], axis=1).values
Y_test = test_data['AVERAGE_DAILY_USAGE']

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

c = 5
model = Lasso(alpha =1/(2*c),max_iter = 10000)
model.fit(X_train,Y_train)
pred = model.predict(X_test)

#create dataframe for plotting results
results_df = pd.DataFrame({'Actual': Y_test, 'Predicted': pred, 'Date': data['DATE'].iloc[test_data.index[0]:test_data.index[-1]]})
results_df.sort_values(by='Date', inplace=True)

#plots
plt.figure(figsize=(15, 7))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Bike Usage',fontsize = 18)
plt.title('Actual vs Predicted Bike Usage - Pandemic Period',fontsize = 20)
plt.legend(fontsize=12,loc='upper right')
plt.show()

plt.figure(figsize=(15, 7))
plt.plot(totals_df['Date'], totals_df['usage'], label='Actual')
plt.plot(results_df['Date'],results_df['Predicted'],label = 'Predicted')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Bike Usage',fontsize = 18)
plt.title('Actual vs Predicted Bike Usage',fontsize = 20)
plt.legend(fontsize=12,loc='upper right')
plt.show()


#POST PANDEMIC 
data_post =  pd.read_csv("post_pandemic.csv")
data_post['DATE'] = pd.to_datetime(data_post['DATE'])

totals_df_post = pd.DataFrame({'Date':data_post['DATE'],'usage':data_post['AVERAGE_DAILY_USAGE']})

plt.figure(figsize=(15, 7))
plt.plot(totals_df_post['Date'], totals_df_post['usage'], label='Actual Usage')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Bike Usage',fontsize=18)
plt.title('Bike Usage',fontsize=20)
plt.legend()
plt.show()
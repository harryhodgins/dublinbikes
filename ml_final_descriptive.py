import pandas as pd
import matplotlib.pyplot as plt

#POST PANDEMIC 
data_post =  pd.read_csv("post_pandemic.csv")
data_post['DATE'] = pd.to_datetime(data_post['DATE'])

totals_df_post = pd.DataFrame({'Date':data_post['DATE'],'usage':data_post['AVERAGE_DAILY_USAGE']})

plt.figure(figsize=(15, 7))
plt.plot(totals_df_post['Date'], totals_df_post['usage'], label='Actual Usage')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Bike Usage',fontsize=18)
plt.title('Bike Usage - 1/07/2018-30/11/2023',fontsize=20)
plt.legend()
plt.show()


#segmenting into periods
pre_pandemic_end = pd.to_datetime("2020-01-31")
pandemic_start = pd.to_datetime("2020-02-01")
pandemic_end = pd.to_datetime("2022-01-31")
post_pandemic_start = pd.to_datetime("2022-02-01")

pre_pandemic = data_post[data_post['DATE'] <= pre_pandemic_end]
pandemic = data_post[(data_post['DATE'] >= pandemic_start) & (data_post['DATE'] <= pandemic_end)]
post_pandemic = data_post[data_post['DATE'] >= post_pandemic_start]

#avg usage for each period
pre_pandemic_avg = pre_pandemic['AVERAGE_DAILY_USAGE'].mean()
pandemic_avg = pandemic['AVERAGE_DAILY_USAGE'].mean()
post_pandemic_avg = post_pandemic['AVERAGE_DAILY_USAGE'].mean()

pre_pandemic_std= pre_pandemic['AVERAGE_DAILY_USAGE'].std()
pandemic_std = pandemic['AVERAGE_DAILY_USAGE'].std()
post_pandemic_std = post_pandemic['AVERAGE_DAILY_USAGE'].std()

print("---MEAN USAGE---")
print("pre pandemic : %s"%pre_pandemic_avg)
print("pandemic : %s"%pandemic_avg)
print("post pandemic : %s"%post_pandemic_avg)

print("---STANDARD DEVIATION---")
print("pre pandemic : %s"%pre_pandemic_std)
print("pandemic : %s"%pandemic_std)
print("post pandemic : %s"%post_pandemic_std)

# Define the date threshold (March 1st)
date_threshold = pd.to_datetime('2020-03-01')
# Filter the DataFrame to keep only rows before March 1st
train_data = data_post[data_post['DATE'] < date_threshold]
test_data = data_post[data_post['DATE'] >= date_threshold]

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

# Create a DataFrame with actual and predicted values and dates
results_df = pd.DataFrame({'Actual': Y_test, 'Predicted': pred, 'Date': data_post['DATE'].iloc[test_data.index[0]:test_data.index[-1]]})

# Sort the DataFrame by date to ensure the plot is ordered
results_df.sort_values(by='Date', inplace=True)

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Bike Usage',fontsize = 18)
plt.title('Actual vs Predicted Bike Usage - Pandemic Period',fontsize = 20)
plt.legend(fontsize=12,loc='upper right')
plt.show()

plt.figure(figsize=(15, 7))
plt.plot(totals_df_post['Date'], totals_df_post['usage'], label='Actual')
plt.plot(results_df['Date'],results_df['Predicted'],label = 'Predicted')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Bike Usage',fontsize = 18)
plt.title('Actual vs Predicted Bike Usage',fontsize = 20)
plt.legend(fontsize=12,loc='upper right')
plt.show()

#segmenting into periods
pre_pandemic_pred = results_df[results_df['Date'] <= pre_pandemic_end]
pandemic_pred = results_df[(results_df['Date'] >= pandemic_start) & (results_df['Date'] <= pandemic_end)]
post_pandemic_pred = results_df[results_df['Date'] >= post_pandemic_start]

#avg usage for each period
pre_pandemic_pred_mean = pre_pandemic_pred['Predicted'].mean()
pandemic_pred_mean = pandemic_pred['Predicted'].mean()
post_pandemic_pred_mean = post_pandemic_pred['Predicted'].mean()

pre_pandemic_pred_std = pre_pandemic_pred['Predicted'].std()
pandemic_pred_std = pandemic_pred['Predicted'].std()
post_pandemic_pred_std = post_pandemic_pred['Predicted'].std()

print("---MEAN USAGE---")
print("pre pandemic : %s" % pre_pandemic_pred_mean)
print("pandemic : %s" % pandemic_pred_mean)
print("post pandemic : %s" % post_pandemic_pred_mean)

print("---STANDARD DEVIATION---")
print("pre pandemic : %s" % pre_pandemic_pred_std)
print("pandemic : %s" % pandemic_pred_std)
print("post pandemic : %s" % post_pandemic_pred_std)

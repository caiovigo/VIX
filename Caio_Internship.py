# -*- coding: utf-8 -*-
"""
@author: Caio Laptop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Importing the data
data=pd.read_csv('C:/Users/Caio Laptop/OneDrive - The University of Kansas/Documents/PhD/12. Summer Applications/03. Summer 2019/' \
                 '36. Arc Stone Capital Management - Quantitative Research Intern/Internship_Exercise/data.csv')

Expiry_Dates=pd.read_csv('C:/Users/Caio Laptop/OneDrive - The University of Kansas/Documents/PhD/12. Summer Applications/03. Summer 2019/' \
                 '36. Arc Stone Capital Management - Quantitative Research Intern/Internship_Exercise/Expiry_Dates.csv', header=None)


#Looking my data
print("Shape: %s" % str(data.shape))
print("------------------------------------------------------------------")
print("Column names: %s" % str(data.columns))
print("------------------------------------------------------------------")
str(data.info())
print('------------------------------------------------------------------')
print(data.describe())
print('------------------------------------------------------------------')
print(data.dtypes)
print('------------------------------------------------------------------')
data.head()


print("Shape: %s" % str(Expiry_Dates.shape))
print("------------------------------------------------------------------")
print("Column names: %s" % str(Expiry_Dates.columns))
print("------------------------------------------------------------------")
str(Expiry_Dates.info())
print('------------------------------------------------------------------')
print(Expiry_Dates.describe())
print('------------------------------------------------------------------')
print(Expiry_Dates.dtypes)
print('------------------------------------------------------------------')
Expiry_Dates.head()

#Defining column names
Expiry_Dates.columns = ['ticker_expiration','date_expiration']

# All tickers
tickers=data['m_localSymbol'].unique()
print(tickers)

# Verifying the type of dates I have in my 'data' dataset
data['date'].dtype
data['date'] = data['date'].values.astype('<M8[D]')

# Defining index of 'data' to be the dates
data.index = data['date']

#Creating new DataFrames for 'VIX','SPX' and VIX Futures
data_VIX = data.loc[data['m_localSymbol']=='VIX']
data_SPX = data.loc[data['m_localSymbol']=='SPX']
data_VXF6 = data.loc[data['m_localSymbol']=='VXF6']
data_VXF7 = data.loc[data['m_localSymbol']=='VXF7']
data_VXF8 = data.loc[data['m_localSymbol']=='VXF8']
data_VXG6 = data.loc[data['m_localSymbol']=='VXG6']
data_VXG7 = data.loc[data['m_localSymbol']=='VXG7']
data_VXG8 = data.loc[data['m_localSymbol']=='VXG8']
data_VXH6 = data.loc[data['m_localSymbol']=='VXH6']
data_VXH7 = data.loc[data['m_localSymbol']=='VXH7']
data_VXH8 = data.loc[data['m_localSymbol']=='VXH8']
data_VXJ6 = data.loc[data['m_localSymbol']=='VXJ6']
data_VXJ7 = data.loc[data['m_localSymbol']=='VXJ7']
data_VXJ8 = data.loc[data['m_localSymbol']=='VXJ8']
data_VXK6 = data.loc[data['m_localSymbol']=='VXK6']
data_VXK7 = data.loc[data['m_localSymbol']=='VXK7']
data_VXK8 = data.loc[data['m_localSymbol']=='VXK8']
data_VXM6 = data.loc[data['m_localSymbol']=='VXM6']
data_VXM7 = data.loc[data['m_localSymbol']=='VXM7']
data_VXM8 = data.loc[data['m_localSymbol']=='VXM8']
data_VXN6 = data.loc[data['m_localSymbol']=='VXN6']
data_VXN7 = data.loc[data['m_localSymbol']=='VXN7']
data_VXN8 = data.loc[data['m_localSymbol']=='VXN8']
data_VXQ6 = data.loc[data['m_localSymbol']=='VXQ6']
data_VXQ7 = data.loc[data['m_localSymbol']=='VXQ7']
data_VXU6 = data.loc[data['m_localSymbol']=='VXU6']
data_VXU7 = data.loc[data['m_localSymbol']=='VXU7']
data_VXV6 = data.loc[data['m_localSymbol']=='VXV6']
data_VXV7 = data.loc[data['m_localSymbol']=='VXV7']
data_VXX6 = data.loc[data['m_localSymbol']=='VXX6']
data_VXX7 = data.loc[data['m_localSymbol']=='VXX7']
data_VXZ6 = data.loc[data['m_localSymbol']=='VXZ6']
data_VXZ7 = data.loc[data['m_localSymbol']=='VXZ7']

# Converting all Time Series to Daily
data_SPX.index = pd.to_datetime(data_SPX['date'])
data_SPX = data_SPX.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_SPX = data_SPX.dropna()

data_VIX.index = pd.to_datetime(data_VIX['date'])
data_VIX = data_VIX.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VIX = data_VIX.dropna()

data_VXQ6.index = pd.to_datetime(data_VXQ6['date'])
data_VXQ6 = data_VXQ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXQ6 = data_VXQ6.dropna()

data_VXK6.index = pd.to_datetime(data_VXK6['date'])
data_VXK6 = data_VXK6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXK6 = data_VXK6.dropna()

data_VXF6.index = pd.to_datetime(data_VXF6['date'])
data_VXF6 = data_VXF6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXF6 = data_VXF6.dropna()

data_VXM6.index = pd.to_datetime(data_VXM6['date'])
data_VXM6 = data_VXM6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXM6 = data_VXM6.dropna()

data_VXH6.index = pd.to_datetime(data_VXH6['date'])
data_VXH6 = data_VXH6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXH6 = data_VXH6.dropna()

data_VXN6.index = pd.to_datetime(data_VXN6['date'])
data_VXN6 = data_VXN6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXN6 = data_VXN6.dropna()

data_VXG6.index = pd.to_datetime(data_VXG6['date'])
data_VXG6 = data_VXG6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXG6 = data_VXG6.dropna()

data_VXU6.index = pd.to_datetime(data_VXU6['date'])
data_VXU6 = data_VXU6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXU6 = data_VXU6.dropna()

data_VXJ6.index = pd.to_datetime(data_VXJ6['date'])
data_VXJ6 = data_VXJ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXJ6 = data_VXJ6.dropna()

data_VXV6.index = pd.to_datetime(data_VXV6['date'])
data_VXV6 = data_VXV6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXV6 = data_VXV6.dropna()

data_VXX6.index = pd.to_datetime(data_VXX6['date'])
data_VXX6 = data_VXX6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXX6 = data_VXX6.dropna()

data_VXZ6.index = pd.to_datetime(data_VXZ6['date'])
data_VXZ6 = data_VXZ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXZ6 = data_VXZ6.dropna()

data_VXF7.index = pd.to_datetime(data_VXF7['date'])
data_VXF7 = data_VXF7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXF7 = data_VXF7.dropna()

data_VXG7.index = pd.to_datetime(data_VXG7['date'])
data_VXG7 = data_VXG7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXG7 = data_VXG7.dropna()

data_VXH7.index = pd.to_datetime(data_VXH7['date'])
data_VXH7 = data_VXH7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXH7 = data_VXH7.dropna()

data_VXJ7.index = pd.to_datetime(data_VXJ7['date'])
data_VXJ7 = data_VXJ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXJ7 = data_VXJ7.dropna()

data_VXK7.index = pd.to_datetime(data_VXK7['date'])
data_VXK7 = data_VXK7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXK7 = data_VXK7.dropna()

data_VXM7.index = pd.to_datetime(data_VXM7['date'])
data_VXM7 = data_VXM7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXM7 = data_VXM7.dropna()

data_VXN7.index = pd.to_datetime(data_VXN7['date'])
data_VXN7 = data_VXN7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXN7 = data_VXN7.dropna()

data_VXQ7.index = pd.to_datetime(data_VXQ7['date'])
data_VXQ7 = data_VXQ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXQ7 = data_VXQ7.dropna()

data_VXU7.index = pd.to_datetime(data_VXU7['date'])
data_VXU7 = data_VXU7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXU7 = data_VXU7.dropna()

data_VXV7.index = pd.to_datetime(data_VXV7['date'])
data_VXV7 = data_VXV7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXV7 = data_VXV7.dropna()

data_VXX7.index = pd.to_datetime(data_VXX7['date'])
data_VXX7 = data_VXX7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXX7 = data_VXX7.dropna()

data_VXZ7.index = pd.to_datetime(data_VXZ7['date'])
data_VXZ7 = data_VXZ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXZ7 = data_VXZ7.dropna()

data_VXF8.index = pd.to_datetime(data_VXF8['date'])
data_VXF8 = data_VXF8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXF8 = data_VXF8.dropna()

data_VXG8.index = pd.to_datetime(data_VXG8['date'])
data_VXG8 = data_VXG8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXG8 = data_VXG8.dropna()

data_VXH8.index = pd.to_datetime(data_VXH8['date'])
data_VXH8 = data_VXH8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXH8 = data_VXH8.dropna()

data_VXJ8.index = pd.to_datetime(data_VXJ8['date'])
data_VXJ8 = data_VXJ8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXJ8 = data_VXJ8.dropna()

data_VXK8.index = pd.to_datetime(data_VXK8['date'])
data_VXK8 = data_VXK8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXK8 = data_VXK8.dropna()

data_VXM8.index = pd.to_datetime(data_VXM8['date'])
data_VXM8 = data_VXM8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXM8 = data_VXM8.dropna()

data_VXN8.index = pd.to_datetime(data_VXN8['date'])
data_VXN8 = data_VXN8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
data_VXN8 = data_VXN8.dropna()

# Creating lists object with all the indexes (SPX, VIX and VIX Futures) - they will be useful later on
data_tickers_list= [data_VIX, data_VXF6, data_VXF7, data_VXF8,
                    data_VXG6, data_VXG7, data_VXG8, data_VXH6,
                    data_VXH7, data_VXH8, data_VXJ6, data_VXJ7,
                    data_VXJ8, data_VXK6, data_VXK7, data_VXK8,
                    data_VXM6, data_VXM7, data_VXM8, data_VXN6,
                    data_VXN7, data_VXN8, data_VXQ6, data_VXQ7,
                    data_VXU6, data_VXU7, data_VXV6, data_VXV7,
                    data_VXX6, data_VXX7, data_VXZ6, data_VXZ7]

data_tickers_list_names = [ 'data_VIX', 'data_VXF6', 'data_VXF7', 'data_VXF8',
                            'data_VXG6', 'data_VXG7', 'data_VXG8', 'data_VXH6',
                            'data_VXH7', 'data_VXH8', 'data_VXJ6', 'data_VXJ7',
                            'data_VXJ8', 'data_VXK6', 'data_VXK7', 'data_VXK8',
                            'data_VXM6', 'data_VXM7', 'data_VXM8', 'data_VXN6',
                            'data_VXN7', 'data_VXN8', 'data_VXQ6', 'data_VXQ7',
                            'data_VXU6', 'data_VXU7', 'data_VXV6', 'data_VXV7',
                            'data_VXX6', 'data_VXX7', 'data_VXZ6', 'data_VXZ7']

# Plotting
# Let's take a look on the plots of all indexes over time

#(1) A panel with plots - each plot represent one index (SPX, VIX or VIX Futures) over time
#plt.style.use('fivethirtyeight')
plt.style.use('default')

fig, axs = plt.subplots(11, 3, sharey=False, sharex=False, tight_layout=False, figsize=(17,11*3.5))
axs[0,0].plot(data_SPX.index, data_SPX['close'], 'b-', label='SPX Close')
axs[0,1].plot(data_VIX.index, data_VIX['close'], 'b-', label='VIX Close')
axs[0,2].plot(data_VXQ6.index, data_VXQ6['close'], 'b-', label='VXQ6 Close')

axs[1,0].plot(data_VXK6.index, data_VXK6['close'], 'b-', label='VXK6 Close')
axs[1,1].plot(data_VXF6.index, data_VXF6['close'], 'b-', label='VXF6 Close')
axs[1,2].plot(data_VXM6.index, data_VXM6['close'], 'b-', label='VXM6 Close')

axs[2,0].plot(data_VXH6.index, data_VXH6['close'], 'b-', label='VXH6 Close')
axs[2,1].plot(data_VXN6.index, data_VXN6['close'], 'b-', label='VXN6 Close')
axs[2,2].plot(data_VXG6.index, data_VXG6['close'], 'b-', label='VXG6 Close')

axs[3,0].plot(data_VXU6.index, data_VXU6['close'], 'b-', label='VXU6 Close')
axs[3,1].plot(data_VXJ6.index, data_VXJ6['close'], 'b-', label='VXJ6 Close')
axs[3,2].plot(data_VXV6.index, data_VXV6['close'], 'b-', label='VXV6 Close')

axs[4,0].plot(data_VXX6.index, data_VXX6['close'], 'b-', label='VXX6 Close')
axs[4,1].plot(data_VXZ6.index, data_VXZ6['close'], 'b-', label='VXZ6 Close')
axs[4,2].plot(data_VXF7.index, data_VXF7['close'], 'b-', label='VXF7 Close')

axs[5,0].plot(data_VXG7.index, data_VXG7['close'], 'b-', label='VXG7 Close')
axs[5,1].plot(data_VXH7.index, data_VXH7['close'], 'b-', label='VXH7 Close')
axs[5,2].plot(data_VXJ7.index, data_VXJ7['close'], 'b-', label='VXJ7 Close')

axs[6,0].plot(data_VXK7.index, data_VXK7['close'], 'b-', label='VXK7 Close')
axs[6,1].plot(data_VXM7.index, data_VXM7['close'], 'b-', label='VXM7 Close')
axs[6,2].plot(data_VXN7.index, data_VXN7['close'], 'b-', label='VXN7 Close')

axs[7,0].plot(data_VXQ7.index, data_VXQ7['close'], 'b-', label='VXQ7 Close')
axs[7,1].plot(data_VXU7.index, data_VXU7['close'], 'b-', label='VXU7 Close')
axs[7,2].plot(data_VXV7.index, data_VXV7['close'], 'b-', label='VXV7 Close')

axs[8,0].plot(data_VXX7.index, data_VXX7['close'], 'b-', label='VXX7 Close')
axs[8,1].plot(data_VXZ7.index, data_VXZ7['close'], 'b-', label='VXZ7 Close')
axs[8,2].plot(data_VXF8.index, data_VXF8['close'], 'b-', label='VXF8 Close')

axs[9,0].plot(data_VXG8.index, data_VXG8['close'], 'b-', label='VXG8 Close')
axs[9,1].plot(data_VXH8.index, data_VXH8['close'], 'b-', label='VXH8 Close')
axs[9,2].plot(data_VXJ8.index, data_VXJ8['close'], 'b-', label='VXJ8 Close')

axs[10,0].plot(data_VXK8.index, data_VXK8['close'], 'b-', label='VXK8 Close')
axs[10,1].plot(data_VXM8.index, data_VXM8['close'], 'b-', label='VXM8 Close')
axs[10,2].plot(data_VXN8.index, data_VXN8['close'], 'b-', label='VXN8 Close')

axs[0,0].legend(loc='best')
axs[0,1].legend(loc='best')
axs[0,2].legend(loc='best')

axs[1,0].legend(loc='best')
axs[1,1].legend(loc='best')
axs[1,2].legend(loc='best')

axs[2,0].legend(loc='best')
axs[2,1].legend(loc='best')
axs[2,2].legend(loc='best')

axs[3,0].legend(loc='best')
axs[3,1].legend(loc='best')
axs[3,2].legend(loc='best')

axs[4,0].legend(loc='best')
axs[4,1].legend(loc='best')
axs[4,2].legend(loc='best')

axs[5,0].legend(loc='best')
axs[5,1].legend(loc='best')
axs[5,2].legend(loc='best')

axs[6,0].legend(loc='best')
axs[6,1].legend(loc='best')
axs[6,2].legend(loc='best')

axs[7,0].legend(loc='best')
axs[7,1].legend(loc='best')
axs[7,2].legend(loc='best')

axs[8,0].legend(loc='best')
axs[8,1].legend(loc='best')
axs[8,2].legend(loc='best')

axs[9,0].legend(loc='best')
axs[9,1].legend(loc='best')
axs[9,2].legend(loc='best')

axs[10,0].legend(loc='best')
axs[10,1].legend(loc='best')
axs[10,2].legend(loc='best')
plt.show()

#(2) Plotting all indexes (SPX, VIX or VIX Futures) in the same plot over time
h=0
plt.style.use('default')
fig, ax1 = plt.subplots(1, sharey=False, tight_layout=False, figsize=(17,5))
ax1.plot(data_VIX.index, data_VIX['close'], label='VIX Close', ls='--')
for df in data_tickers_list:
    ax1.plot(df.index, df['close'], label=data_tickers_list_names[h])
    h=h+1
ax1.legend(loc='upper right', ncol=2, prop={'size': 6})
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(data_SPX.index, data_SPX['close'], label='SPX Close', c=(0,0,0))
ax2.legend(loc='upper center', prop={'size': 8})
plt.show()
    
# Calculating the returns of SPX
daily_return_close = pd.DataFrame(data_SPX['close'].pct_change(1))
daily_return_close.columns = ['returns_close']
data_SPX = pd.merge(data_SPX, daily_return_close, left_index=True,right_index=True)
#data_SPX.iloc[[0], [4]]=0

# Plotting the histogram of the SPX returns
fig, axs = plt.subplots(1, figsize=(5,5))
axs.hist(data_SPX.ix[1:len(data_SPX),'returns_close'], bins=25, color='red')

# Let's define market selloff
# Now, to define what we consider the threshold for a market selloff, we decided to use the 2.5% percentile, using the quantile function. 
#The reason is simple, any threshold would be arbitrary. Doing so, we guarantee that we will have 2.5% of datapoints for our estimations. 
#Easily, we can change this threshold and re-run all the code to get some different results.

data_SPX['returns_close'].quantile(0.025)
# The 2.5% quantile give us -0.01503559917857245. Therefore, I will set a market selloff=-0.015, i.e., whenever the SPX falls 1.5% or more in a single day
mkt_selloff = -0.015

market_selloff_days = data_SPX[data_SPX['returns_close']<=-0.015]
market_selloff_days.index
# Therefore, we have only 12 days in our sample in which the SPX fell 1.5% or more in a single day


# Calculating the daily returns of all VIX and VIX Futures
data_VIX = pd.merge(data_VIX, pd.DataFrame(data_VIX['close'].pct_change(1)), left_index=True, right_index=True)
data_VXF6 = pd.merge(data_VXF6, pd.DataFrame(data_VXF6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXF7 = pd.merge(data_VXF7, pd.DataFrame(data_VXF7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXF8 = pd.merge(data_VXF8, pd.DataFrame(data_VXF8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXG6 = pd.merge(data_VXG6, pd.DataFrame(data_VXG6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXG7 = pd.merge(data_VXG7, pd.DataFrame(data_VXG7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXG8 = pd.merge(data_VXG8, pd.DataFrame(data_VXG8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXH6 = pd.merge(data_VXH6, pd.DataFrame(data_VXH6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXH7 = pd.merge(data_VXH7, pd.DataFrame(data_VXH7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXH8 = pd.merge(data_VXH8, pd.DataFrame(data_VXH8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXJ6 = pd.merge(data_VXJ6, pd.DataFrame(data_VXJ6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXJ7 = pd.merge(data_VXJ7, pd.DataFrame(data_VXJ7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXJ8 = pd.merge(data_VXJ8, pd.DataFrame(data_VXJ8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXK6 = pd.merge(data_VXK6, pd.DataFrame(data_VXK6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXK7 = pd.merge(data_VXK7, pd.DataFrame(data_VXK7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXK8 = pd.merge(data_VXK8, pd.DataFrame(data_VXK8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXM6 = pd.merge(data_VXM6, pd.DataFrame(data_VXM6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXM7 = pd.merge(data_VXM7, pd.DataFrame(data_VXM7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXM8 = pd.merge(data_VXM8, pd.DataFrame(data_VXM8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXN6 = pd.merge(data_VXN6, pd.DataFrame(data_VXN6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXN7 = pd.merge(data_VXN7, pd.DataFrame(data_VXN7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXN8 = pd.merge(data_VXN8, pd.DataFrame(data_VXN8['close'].pct_change(1)), left_index=True, right_index=True)
data_VXQ6 = pd.merge(data_VXQ6, pd.DataFrame(data_VXQ6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXQ7 = pd.merge(data_VXQ7, pd.DataFrame(data_VXQ7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXU6 = pd.merge(data_VXU6, pd.DataFrame(data_VXU6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXU7 = pd.merge(data_VXU7, pd.DataFrame(data_VXU7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXV6 = pd.merge(data_VXV6, pd.DataFrame(data_VXV6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXV7 = pd.merge(data_VXV7, pd.DataFrame(data_VXV7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXX6 = pd.merge(data_VXX6, pd.DataFrame(data_VXX6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXX7 = pd.merge(data_VXX7, pd.DataFrame(data_VXX7['close'].pct_change(1)), left_index=True, right_index=True)
data_VXZ6 = pd.merge(data_VXZ6, pd.DataFrame(data_VXZ6['close'].pct_change(1)), left_index=True, right_index=True)
data_VXZ7 = pd.merge(data_VXZ7, pd.DataFrame(data_VXZ7['close'].pct_change(1)), left_index=True, right_index=True)

# Updating data_tickers_list
data_tickers_list= [data_VIX, data_VXF6, data_VXF7, data_VXF8,
                    data_VXG6, data_VXG7, data_VXG8, data_VXH6,
                    data_VXH7, data_VXH8, data_VXJ6, data_VXJ7,
                    data_VXJ8, data_VXK6, data_VXK7, data_VXK8,
                    data_VXM6, data_VXM7, data_VXM8, data_VXN6,
                    data_VXN7, data_VXN8, data_VXQ6, data_VXQ7,
                    data_VXU6, data_VXU7, data_VXV6, data_VXV7,
                    data_VXX6, data_VXX7, data_VXZ6, data_VXZ7]

# Plotting the SPX returns on x-axis and VIX & VIX Futures returns on the y-axis for the days in which the SPX had a selloff (threshhold defined above)
i=0
j=0
h=0
plt.style.use('default')
fig, axs = plt.subplots(11, 3, sharey=False, tight_layout=True, squeeze=False, figsize=(10,11*2))
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days.index)]['close_y']
    Y = Y.dropna()
    X = data_SPX[data_SPX.index.isin(Y.index)]['returns_close']
    axs[i,j].scatter(X,Y, c='red')
    axs[i,j].set(title=data_tickers_list_names[h],
                 ylabel=data_tickers_list_names[h],
                 xlabel='SPX returns')
    j=j+1
    h=h+1
    if j == 3:
        j=0
        i=i+1
"""
Notice that not all VIX Futures have data for the market selloff days. That's why some of the plots are blank.
Notice as well that, even though the sample size is not that large for some VIX Futures, we still can - eye inspecting - 
see a relationship between SPX and VIX / VIX Futures. Looks like that we have a negative linear association for the same day of market selloffs.
"""

# Plotting the histograms of VIX & VIX Futures returns for the days in which the SPX had a selloff (threshhold defined above)
i=0
j=0
h=0
plt.style.use('default')
fig, axs = plt.subplots(11, 3, sharey=False, tight_layout=True, squeeze=False, figsize=(10,11*2))
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days.index)]['close_y']
    Y = Y.dropna()
    axs[i,j].hist(Y, color='purple')
    axs[i,j].set(title=data_tickers_list_names[h])
    j=j+1
    h=h+1
    if j == 3:
        j=0
        i=i+1
        
# Running OLS cross-sections regressions to  to determine how the VIX futures curve behaves in market selloff (threshhold defined above).
# Thus, the dependent variable is SPX returns during the selloff and the independent variable is
# VIX & VIX future.
cols = ['Correlation', 'Intercept', 'Slope-beta_1', 'Intercept [p-value]', 'Slope-beta_1 [p-value]', 'R2', 'F-test p-value']
lst = []       
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days.index)]['close_y']
    Y = Y.dropna()
    X = data_SPX[data_SPX.index.isin(Y.index)]['returns_close']
    my_corr = np.corrcoef(X,Y)[0,1]
    if len(X)>=2:
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
#        print(results.summary())
    
    lst.append([my_corr, results.params[0], results.params[1], 
                results.pvalues[0], results.pvalues[1], 
                results.rsquared, results.f_pvalue])

Regression_results_extended = pd.DataFrame(lst, columns=cols)
Regression_results_extended.index =data_tickers_list_names
print(Regression_results_extended) 
        

# Since we don't have that many datapoints for selloff days and also because we want to analyze how the SPX selloff affect VIX and VIX futures in the following days,
#we will consider up to 5 days of trading after the selloff
        
market_selloff_days_plus1 = market_selloff_days.index + pd.DateOffset(1)
market_selloff_days_plus1 = pd.DataFrame(data_SPX[data_SPX.index.isin(market_selloff_days_plus1)]['returns_close'])
       
market_selloff_days_plus2 = market_selloff_days.index + pd.DateOffset(2)
market_selloff_days_plus2 = pd.DataFrame(data_SPX[data_SPX.index.isin(market_selloff_days_plus2)]['returns_close'])
   
market_selloff_days_plus3 = market_selloff_days.index + pd.DateOffset(3)
market_selloff_days_plus3 = pd.DataFrame(data_SPX[data_SPX.index.isin(market_selloff_days_plus3)]['returns_close'])
   
market_selloff_days_plus4 = market_selloff_days.index + pd.DateOffset(4)   
market_selloff_days_plus4 = pd.DataFrame(data_SPX[data_SPX.index.isin(market_selloff_days_plus4)]['returns_close'])

market_selloff_days_plus5 = market_selloff_days.index + pd.DateOffset(5)   
market_selloff_days_plus5 = pd.DataFrame(data_SPX[data_SPX.index.isin(market_selloff_days_plus5)]['returns_close'])

market_selloff_days_plus_extended = pd.concat([market_selloff_days_plus1, 
                                               market_selloff_days_plus2,
                                               market_selloff_days_plus3, 
                                               market_selloff_days_plus4,
                                               market_selloff_days_plus5])

# Plotting the SPX returns on x-axis and VIX & VIX Futures returns on the y-axis for the days in which the SPX had a selloff (threshhold defined above)
i=0
j=0
h=0
plt.style.use('default')
fig, axs = plt.subplots(11, 3, sharey=False, tight_layout=True, squeeze=False, figsize=(10,11*2))
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days_plus_extended.index)]['close_y']
    Y = Y.dropna()
    X = data_SPX[data_SPX.index.isin(Y.index)]['returns_close']
    axs[i,j].scatter(X,Y, c='red')
    axs[i,j].set(title=data_tickers_list_names[h],
                 ylabel=data_tickers_list_names[h],
                 xlabel='SPX returns')
    j=j+1
    h=h+1
    if j == 3:
        j=0
        i=i+1

# Plotting the histograms of VIX & VIX Futures returns for the days in which the SPX had a selloff (threshhold defined above)
i=0
j=0
h=0
plt.style.use('default')
fig, axs = plt.subplots(11, 3, sharey=False, tight_layout=True, squeeze=False, figsize=(10,11*2))
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days_plus_extended.index)]['close_y']
    Y = Y.dropna()
    axs[i,j].hist(Y, color='purple')
    axs[i,j].set(title=data_tickers_list_names[h])
    j=j+1
    h=h+1
    if j == 3:
        j=0
        i=i+1

# Running OLS cross-sections regressions to  to determine how the VIX futures curve behaves in market selloff (threshhold defined above).
# Again, the dependent variable is SPX returns during the selloff (extend up to five days of the fall) and the independent variable is
# VIX & VIX future.
cols = ['Correlation', 'Intercept', 'Slope-beta_1', 'Intercept [p-value]', 'Slope-beta_1 [p-value]', 'R2', 'F-test p-value']
lst = []       
for df in data_tickers_list:
    Y = df[df.index.isin(market_selloff_days_plus_extended.index)]['close_y']
    Y = Y.dropna()
    X = data_SPX[data_SPX.index.isin(Y.index)]['returns_close']
    my_corr = np.corrcoef(X,Y)[0,1]
    if len(X)>=2:
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
#        print(results.summary())
    
    lst.append([my_corr, results.params[0], results.params[1], 
                results.pvalues[0], results.pvalues[1], 
                results.rsquared, results.f_pvalue])

Regression_results_extended = pd.DataFrame(lst, columns=cols)
Regression_results_extended.index =data_tickers_list_names
print(Regression_results_extended) 


# Plotting the VIX returns on x-axis and VIX Futures returns on the y-axis for the days in which the SPX had a selloff (threshhold defined above)
i=0
j=0
h=0
plt.style.use('default')
fig, axs = plt.subplots(11, 3, sharey=False, tight_layout=True, squeeze=False, figsize=(10,11*2))
for df in data_tickers_list[1:(len(data_tickers_list)+1)]:
    Y = df[df.index.isin(market_selloff_days_plus_extended.index)]['close_y']
    Y = Y.dropna()
    X = data_VIX[data_VIX.index.isin(Y.index)]['close_y']
    axs[i,j].scatter(X,Y, c='red')
    axs[i,j].set(title=data_tickers_list_names[h+1],
                 ylabel=data_tickers_list_names[h+1],
                 xlabel='VIX returns')
    j=j+1
    h=h+1
    if j == 3:
        j=0
        i=i+1

# Running OLS cross-sections regressions to  to determine how the VIX futures curve behaves in market selloff (threshhold defined above).
# Thus, the dependent variable is SPX returns during the selloff (extend up to five days of the fall) and the independent variable is
# VIX & VIX future.
cols = ['Correlation', 'Intercept', 'Slope-beta_1', 'Intercept [p-value]', 'Slope-beta_1 [p-value]', 'R2', 'F-test p-value']
lst = []       
for df in data_tickers_list[1:(len(data_tickers_list)+1)]:
    Y = df[df.index.isin(market_selloff_days_plus_extended.index)]['close_y']
    Y = Y.dropna()
    X = data_VIX[data_VIX.index.isin(Y.index)]['close_y']
    my_corr = np.corrcoef(X,Y)[0,1]
    if len(X)>=2:
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
    #   print(results.summary())
    
    lst.append([my_corr, results.params[0], results.params[1], 
                results.pvalues[0], results.pvalues[1], 
                results.rsquared, results.f_pvalue])

Regression_results_extended = pd.DataFrame(lst, columns=cols)
Regression_results_extended.index =data_tickers_list_names[1:(len(data_tickers_list)+1)]
print(Regression_results_extended)



from scipy import stats

sm.graphics.tsa.plot_acf(data_VIX['close_x'], lags=15)
sm.graphics.tsa.plot_pacf(data_VIX['close_x'], lags=15)
sm.tsa.stattools.pacf_ols(data_VIX['close_x'], nlags=15)

arma_mod = sm.tsa.ARMA(data_VIX['close_x'], (2,0)).fit(disp=False)
print(arma_mod.params)

fig, axs = plt.subplots(1, figsize=(6,4))
arma_mod.resid.plot(ax=axs)

resid = arma_mod.resid
stats.normaltest(resid)
fig, axs = plt.subplots(1, figsize=(6,4))
sm.graphics.qqplot(resid, line='q', ax=axs, fit=True)

#---------------------------------------------------------------------------------------------------------------------
# SKETCHES
#---------------------------------------------------------------------------------------------------------------------


#all_tickers = pd.concat([data_VXQ6, data_VXQ6])
#corr = df.corr()
#corr.style.background_gradient()
#
#VIX_test=data.groupby(by='m_localSymbol').get_group('VIX')
#dates=pd.to_datetime(data_VXF6['date'], infer_datetime_format=True)
#datetime.date.day(data_VXF6['date'])
#
#test=data_VXF6.groupby(pd.to_datetime(data_VXF6['date'], format='%Y/%m/%d')).agg({'open':  lambda x: x.first,
#                                                                                  'high':  lambda x: x.max(),
#                                                                                  'low':   lambda x: x.min(),                                                                                  ,
#                                                                                  'close': lambda x: x[-1]})

#---------------------------------------------------------------------------------------------------------------------
#data_SPX = data_SPX.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_SPX = data_SPX.dropna()
#
#data_VIX = data_VIX.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VIX = data_VIX.dropna()
#
#data_VXQ6 = data_VXQ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXQ6 = data_VXQ6.dropna()
#
#data_VXK6 = data_VXK6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXK6 = data_VXK6.dropna()
#
#data_VXF6 = data_VXF6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXF6 = data_VXF6.dropna()
#
#data_VXM6 = data_VXM6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXM6 = data_VXM6.dropna()
#
#data_VXH6 = data_VXH6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXH6 = data_VXH6.dropna()
#
#data_VXN6 = data_VXN6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXN6 = data_VXN6.dropna()
#
#data_VXG6 = data_VXG6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXG6 = data_VXG6.dropna()
#
#data_VXU6 = data_VXU6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXU6 = data_VXU6.dropna()
#
#data_VXJ6 = data_VXJ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXJ6 = data_VXJ6.dropna()
#
#data_VXV6 = data_VXV6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXV6 = data_VXV6.dropna()
#
#data_VXX6 = data_VXX6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXX6 = data_VXX6.dropna()
#
#data_VXZ6 = data_VXZ6.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXZ6 = data_VXZ6.dropna()
#
#data_VXF7 = data_VXF7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXF7 = data_VXF7.dropna()
#
#data_VXG7 = data_VXG7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXG7 = data_VXG7.dropna()
#
#data_VXH7 = data_VXH7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXH7 = data_VXH7.dropna()
#
#data_VXJ7 = data_VXJ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXJ7 = data_VXJ7.dropna()
#
#data_VXK7 = data_VXK7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXK7 = data_VXK7.dropna()
#
#data_VXM7 = data_VXM7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXM7 = data_VXM7.dropna()
#
#data_VXN7 = data_VXN7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXN7 = data_VXN7.dropna()
#
#data_VXQ7 = data_VXQ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXQ7 = data_VXQ7.dropna()
#
#data_VXU7 = data_VXU7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXU7 = data_VXU7.dropna()
#
#data_VXV7 = data_VXV7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXV7 = data_VXV7.dropna()
#
#data_VXX7 = data_VXX7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXX7 = data_VXX7.dropna()
#
#data_VXZ7 = data_VXZ7.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXZ7 = data_VXZ7.dropna()
#
#data_VXF8 = data_VXF8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXF8 = data_VXF8.dropna()
#
#data_VXG8 = data_VXG8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXG8 = data_VXG8.dropna()
#
#data_VXH8 = data_VXH8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXH8 = data_VXH8.dropna()
#
#data_VXJ8 = data_VXJ8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXJ8 = data_VXJ8.dropna()
#
#data_VXK8 = data_VXK8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXK8 = data_VXK8.dropna()
#
#data_VXM8 = data_VXM8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXM8 = data_VXM8.dropna()
#
#data_VXN8 = data_VXN8.resample('D').apply({'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXN8 = data_VXN8.dropna()

#data_VXN8x = data_VXN8.resample('D', how={'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'})
#data_VXN8x = data_VXN8x.dropna()

#---------------------------------------------------------------------------------------------------------------------
#plt.style.use('default')
#fig, ax1 = plt.subplots(1, sharey=False, tight_layout=False, figsize=(17,5))
#ax1.plot(data_VIX.index, data_VIX['close'], label='VIX Close', ls='--')
#ax1.plot(data_VXQ6.index, data_VXQ6['close'], label='VXQ6 Close')
#ax1.plot(data_VXK6.index, data_VXK6['close'], label='VXK6 Close')
#ax1.plot(data_VXF6.index, data_VXF6['close'], label='VXF6 Close')
#ax1.plot(data_VXM6.index, data_VXM6['close'], label='VXM6 Close')
#ax1.plot(data_VXH6.index, data_VXH6['close'], label='VXH6 Close')
#ax1.plot(data_VXN6.index, data_VXN6['close'], label='VXN6 Close')
#ax1.plot(data_VXG6.index, data_VXG6['close'], label='VXG6 Close')
#ax1.plot(data_VXU6.index, data_VXU6['close'], label='VXU6 Close')
#ax1.plot(data_VXJ6.index, data_VXJ6['close'], label='VXJ6 Close')
#ax1.plot(data_VXV6.index, data_VXV6['close'], label='VXV6 Close')
#ax1.plot(data_VXX6.index, data_VXX6['close'], label='VXX6 Close')
#ax1.plot(data_VXZ6.index, data_VXZ6['close'], label='VXZ6 Close')
#ax1.plot(data_VXF7.index, data_VXF7['close'], label='VXF7 Close')
#ax1.plot(data_VXG7.index, data_VXG7['close'], label='VXG7 Close')
#ax1.plot(data_VXH7.index, data_VXH7['close'], label='VXH7 Close')
#ax1.plot(data_VXJ7.index, data_VXJ7['close'], label='VXJ7 Close')
#ax1.plot(data_VXK7.index, data_VXK7['close'], label='VXK7 Close')
#ax1.plot(data_VXM7.index, data_VXM7['close'], label='VXM7 Close')
#ax1.plot(data_VXN7.index, data_VXN7['close'], label='VXN7 Close')
#ax1.plot(data_VXQ7.index, data_VXQ7['close'], label='VXQ7 Close')
#ax1.plot(data_VXU7.index, data_VXU7['close'], label='VXU7 Close')
#ax1.plot(data_VXV7.index, data_VXV7['close'], label='VXV7 Close')
#ax1.plot(data_VXX7.index, data_VXX7['close'], label='VXX7 Close')
#ax1.plot(data_VXZ7.index, data_VXZ7['close'], label='VXZ7 Close')
#ax1.plot(data_VXF8.index, data_VXF8['close'], label='VXF8 Close')
#ax1.plot(data_VXG8.index, data_VXG8['close'], label='VXG8 Close')
#ax1.plot(data_VXH8.index, data_VXH8['close'], label='VXH8 Close')
#ax1.plot(data_VXJ8.index, data_VXJ8['close'], label='VXJ8 Close')
#ax1.plot(data_VXK8.index, data_VXK8['close'], label='VXK8 Close')
#ax1.plot(data_VXM8.index, data_VXM8['close'], label='VXM8 Close')
#ax1.plot(data_VXN8.index, data_VXN8['close'], label='VXN8 Close')
#ax1.legend(loc='upper right', ncol=2, prop={'size': 6})
#ax1.grid(True)
#ax2 = ax1.twinx()
#ax2.plot(data_SPX.index, data_SPX['close'], label='SPX Close', c=(0,0,0))
#ax2.legend(loc='upper center', prop={'size': 8})
#plt.show()
#---------------------------------------------------------------------------------------------------------------------
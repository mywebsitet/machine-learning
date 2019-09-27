
# coding: utf-8

# # Module 2- Plotting in Financial Markets
# 

#    ### Welcome to the Answer notebook for Module 2 ! 
# Make sure that you've submitted the module 1 notebook and unlocked Module 2 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 2.1 
# Load the week2.csv file into a dataframe. What is the type of the Date column? Make sure it is of type datetime64. Convert the Date column to the index of the dataframe.
# Plot the closing price of each of the days for the entire time frame to get an idea of what the general outlook of the stock is.
# 
# >Look out for drastic changes in this stock, you have the exact date when these took place, try to fetch the news for this day of this stock
# 
# >This would be helpful if we are to train our model to take NLP inputs.

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#And solve the query
import os
import datetime
from functools import partial


# In[7]:


data=pd.read_csv("week2.csv")
#del data['unnamed:0']
duplicate_data = data


# In[8]:


data.head()


# In[9]:


data.Date.dtype


# In[10]:


data['Date']= pd.to_datetime(data['Date'])
data.Date.dtype


# In[11]:


data['Close Price'].plot(figsize=(20,8))
plt.legend()


# In[12]:


#2.2
data.head(2)


# In[13]:


plt.figure(figsize=(20,10))
plt.stem(data.Date,data['Day_Perc_Change'])


# In[14]:


#2.3


# In[15]:


plt.figure(figsize=(20,10))
plt.stem(data.Date,data['Total Traded Quantity'])
plt.show()


# In[16]:


plt.figure(figsize=(20,10))
plt.stem(data['Total Traded Quantity'],data['Day_Perc_Change'])
plt.show()
plt.figure(figsize=(20,10))
plt.stem(data['Day_Perc_Change'],data['Total Traded Quantity'])
plt.show()


# In[17]:


plt.figure(figsize=(20,10))
plt.plot(data['Total Traded Quantity'],data['Day_Perc_Change'])
plt.show()
plt.figure(figsize=(20,10))
plt.plot(data['Day_Perc_Change'],data['Total Traded Quantity'])
plt.show()


# In[18]:


from collections import Counter
Trendsare = ['Postive','Negative','Breakout Bull','Breakout Bear','Among top losers','Among top gainers','Slight or No Change','Slight Positive','Slight Negative']
Trend_to_list = data['Trend'].tolist()
counts = Counter(Trend_to_list)
print(counts)


# In[19]:


colors = ['r']
counter=[497]
labels= ['Slight or No change']
plt.figure(figsize=(20,10))
plt.pie(counter, labels=labels,colors=colors,startangle=90, autopct='%.1f%%')
plt.show()


# In[20]:


plt.figure(figsize=(20,10))
data.groupby(['Trend'])['Total Traded Quantity'].mean().plot.bar()


# In[21]:


plt.figure(figsize=(20,10))
data.groupby(['Trend'])['Total Traded Quantity'].median().plot.bar()


# In[22]:


#2.5

plt.figure(figsize=(20,10))
plt.hist(data['Day_Perc_Change'])
plt.show()


# In[23]:


ashoka_data=pd.read_csv("ASHOKA.csv")
bajajelec_data=pd.read_csv("BAJAJELEC.csv")
centuryply_data=pd.read_csv("CENTURYPLY.csv")
idfc_data = pd.read_csv("IDFC.csv")
itdc_data = pd.read_csv("ITDC.csv")


# In[24]:


filter_ashoka_data = ashoka_data[ashoka_data.Series == 'EQ']
filter_ashoka_data


# In[25]:


filter_bajajelec_data = bajajelec_data[bajajelec_data.Series == 'EQ']
filter_bajajelec_data


# In[26]:


filter_centuryply_data = centuryply_data[centuryply_data.Series == 'EQ']
filter_centuryply_data


# In[27]:


filter_idfc_data = idfc_data[idfc_data.Series == 'EQ']
filter_idfc_data


# In[28]:


filter_itdc_data = itdc_data[itdc_data.Series == 'EQ']
filter_itdc_data


# In[29]:


columns = ['ASHOKA','BAJAJELEC','CENTURYPLY','IDFC','ITDC']
close_prices_dataFrame = pd.DataFrame(columns = columns)
close_prices_dataFrame['ASHOKA'] = filter_ashoka_data['Close Price']
close_prices_dataFrame['BAJAJELEC'] = filter_bajajelec_data['Close Price']
close_prices_dataFrame['CENTURYPLY'] = filter_centuryply_data['Close Price']
close_prices_dataFrame['IDFC'] = filter_idfc_data['Close Price']
close_prices_dataFrame['ITDC'] = filter_itdc_data['Close Price']
close_prices_dataFrame.dropna()


# In[30]:


pct_change_dataFrame =close_prices_dataFrame.pct_change().fillna(0)
pct_change_dataFrame.dropna()


# In[31]:


sns.set(color_codes=True)
sns.pairplot(pct_change_dataFrame)


# In[32]:


rolling_avg_idfc = pct_change_dataFrame['IDFC'].rolling(7).mean()
rolling_avg_idfc


# In[33]:


stand_idfc = rolling_avg_idfc.fillna(0).std()
stand_idfc


# In[34]:


currdate=pd.to_datetime(filter_idfc_data["Date"])
currlist=currdate.tolist()
plt.figure(figsize=(20,10))
plt.plot(currlist,rolling_avg_idfc.fillna(0).tolist())
plt.show()


# In[35]:


#2.8

nift_load = pd.read_csv("Nifty50.csv")
nift_load


# In[36]:


nift_close_price = nift_load['Close']
nift_change=nift_close_price.pct_change().fillna(0).rolling(7).mean().fillna(0) 
nift_date=pd.to_datetime(nift_load['Date'])
nift_date=nift_date.tolist()
plt.figure(figsize=(20,10))


itdc_Date = pd.to_datetime(filter_itdc_data['Date'])
itdcLis = itdc_Date.tolist()
itdc_close_price = filter_itdc_data['Close Price']
itdc_change = itdc_close_price.pct_change().fillna(0).rolling(7).mean().fillna(0)
plt.figure(figsize=(20,10))
plt.plot(itdcLis,rolling_avg_idfc.fillna(0).tolist())

plt.title("Volatility of NIFTY with respect to ITDC and IDFC")
plt.plot(nift_date,nift_change.tolist(),label = 'nifty')
plt.plot(currlist,rolling_avg_idfc.fillna(0).tolist(),label = 'idfc')
plt.plot(itdcLis,itdc_change,label = 'itdc')
plt.legend(loc='upper left')
plt.show()


# In[37]:


#2.9
plt.figure(figsize=(20,10))
plt.plot(itdcLis,itdc_change,label = 'itdc')
plt.legend(loc='upper left')
plt.show()


# In[43]:


short_window = 21
long_window = 34
signals = pd.DataFrame(index=filter_idfc_data.index)
signals['signal'] = 0.0
signals['short_mavg'] = filter_idfc_data['Close Price'].rolling(window=short_window, min_periods=1,center=False).mean()
signals['long_mavg'] = filter_idfc_data['Close Price'].rolling(window=long_window,min_periods=1, center=False).mean()

signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0,0.0)

signals['positions'] = signals['signal'].diff()
print(signals)


# In[44]:



fig = plt.figure(figsize=(20,15))

ax1 = fig.add_subplot(111, ylabel='Price')

filter_idfc_data['Close Price'].plot(ax=ax1, color='black', lw=2.)

signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^' , markersize=20,color='g')

ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v' , markersize=20,color='r')

plt.show()


# In[60]:



symbol = 'IDFC'

# read csv file, use date as index and read close as a column
df = pd.read_csv('IDFC.csv'.format(symbol), index_col='Date',
                 parse_dates=True, usecols=['Date', 'Close Price'],
                 na_values='nan')

# rename the column header with symbol name
df = df.rename(columns={'Close Price': symbol})
df.dropna(inplace=True)

# calculate Simple Moving Average with 14 days window
sma = df.rolling(window=14).mean()

# calculate the standar deviation
rstd = df.rolling(window=14).std()

upper_band = sma + 2 * rstd
upper_band = upper_band.rename(columns={symbol: 'upper'})
lower_band = sma - 2 * rstd
lower_band = lower_band.rename(columns={symbol: 'lower'})
df = df.join(upper_band).join(lower_band)
ax = df.plot(title='{} Price and BB'.format(symbol))

ax.set_xlabel('Date')
ax.set_ylabel('SMA and BB')
ax.grid()
plt.show()


# machine-learning

# ### Query 1.1 
# Import the csv file of the stock of your choosing using 'pd.read_csv()' function into a dataframe.
# Shares of a company can be offered in more than one category. The category of a stock is indicated in the ‘Series’ column. If the csv file has data on more than one category, the ‘Date’ column will have repeating values. To avoid repetitions in the date, remove all the rows where 'Series' column is NOT 'EQ'.
# Analyze and understand each column properly.
# You'd find the head(), tail() and describe() functions to be immensely useful for exploration. You're free to carry out any other exploration of your own.

# In[1]:


#The solution code should start right after the query statement, for example -
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#And solve the query
import os
import datetime
from functools import partial


# In[7]:


data=pd.read_csv("TCS.csv")


# In[8]:


print('data')


# In[12]:


#Once the solution of the first query is written, it should immediately be followed by the next query

filtered_data= data[data.Series == 'EQ']

filtered_data


# In[13]:



filtered_data.head()


# In[14]:



filtered_data.tail()


# In[15]:


filtered_data.describe()


# ### Query 1.2
# Calculate the maximum, minimum and mean price for the last 90 days. (price=Closing Price unless stated otherwise)

# In[18]:


filtered_data.tail(90)['Close Price'].max()


# In[20]:


filtered_data.tail(90)['Close Price'].min()


# **This is the expected format of the answer notebook**

# In[21]:


filtered_data.tail(90)['Close Price'].mean()


# In[22]:


filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
filtered_data['Date'].dtype


# In[23]:


filtered_data['Date'].max()-filtered_data['Date'].min()


# In[25]:


data['Month']=pd.DatetimeIndex(data['Date']).month
data['Year'] = pd.DatetimeIndex(data['Date']).year
data['VWAP']= (data['Close Price']*data['Total Traded Quantity']).cumsum() / data['Total Traded Quantity'].fillna(0).cumsum()
data_vwap=data[['Month','Year','VWAP']]
group = data_vwap.groupby(['Month','Year'])


# In[28]:


def avg_price(N):
	return (data['Average Price'].tail(N).sum())/N
print("Average prices for last N days are as follows:")
print("Last 1 week",avg_price(5))
print("Last 2 weeks",avg_price(10))
print("Last 1 month",avg_price(20))
print("Last 3 months",avg_price(60))
print("Last 6 months",avg_price(120))
print("Last 1 year",avg_price(240))

print("Profit/Loss % for N days are as follows:")
def prof_loss(N):
    difference = (data['Close Price'].tail(N).iloc[N-1] - data['Close Price'].tail(N).iloc[0])
    if difference < 0 :
        loss = -(difference)
        loss_percen = (loss/data['Close Price'].tail(N).iloc[N-1])*100
        return loss_percen
    if difference > 0 :
        profit = difference
        profit_percen = (profit/data['Close Price'].tail(N).iloc[N-1])*100
        return profit_percen
print("Loss/Profit percentage for last N days are as follows:")
print("Last 1 week",prof_loss(5))
print("Last 2 weeks",prof_loss(10))
print("Last 1 month",prof_loss(20))
print("Last 3 months",prof_loss(60))
print("Last 6 months",prof_loss(120))
print("Last 1 year",prof_loss(240))


# In[29]:


data['Day_Perc_Change'] = data['Close Price'].pct_change().fillna(0)
data


# In[45]:


#data.loc[(data['Day_Perc_Change']>0.5) & (data['Day_Perc_Change']<1),'Trend'] = 'Slight positive'
data.loc[(data['Day_Perc_Change']>-0.5) & (data['Day_Perc_Change']<0.5),'Trend'] = 'Slight positive'
data.loc[(data['Day_Perc_Change']>0.5) & (data['Day_Perc_Change']<1),'Trend'] = 'Slight positive'
data.loc[(data['Day_Perc_Change']>-1) & (data['Day_Perc_Change']<-0.5),'Trend'] = 'Slight negative'
data.loc[(data['Day_Perc_Change']>1) & (data['Day_Perc_Change']<3),'Trend'] = 'Positive'
data.loc[(data['Day_Perc_Change']>-3) & (data['Day_Perc_Change']<-1),'Trend'] = 'Negative'
data.loc[(data['Day_Perc_Change']>3) & (data['Day_Perc_Change']<7),'Trend'] = 'Among top gainers'
data.loc[(data['Day_Perc_Change']>-7) & (data['Day_Perc_Change']<-3),'Trend'] = 'Among top losers'
data.loc[data['Day_Perc_Change']>7,'Trend'] = 'Bull run'
data.loc[data['Day_Perc_Change']<-7,'Trend'] = 'Bear drop'

data


# In[46]:


data.groupby(data.Trend).mean()['Total Traded Quantity']
data.groupby(data.Trend).median()['Total Traded Quantity']


# In[47]:


data.to_csv("week2.csv")


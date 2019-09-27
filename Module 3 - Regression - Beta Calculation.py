
# coding: utf-8

# # Module 3- Regression & Beta Calculation
# 
# 

#    ### Welcome to the Answer notebook for Module 3 ! 
# Make sure that you've submitted the module 2 notebook and unlocked Module 3 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 3.1 
# Import the file 'gold.csv', which is contains the data about last 2 years price action of Indian(MCX) gold standard. Explore the dataframe. You'd see 2 unique columns - 'Pred' and 'new'.
# 
# One of the 2 columns is a linear combination of the OHLC prices with varying coefficients while the other is a polynomial fucntion of the same inputs. Also, one of the 2 columns is partially filled.
# 
# >Using linear regression, find the coefficients of the inputs and using the same trained model, complete the
#       entire column.
#       
# >Also, try to fit the other column as well using a new linear regression model. Check if the predicitons are 
#       accurate.
#       Mention which column is a linear function and which is a polynomial function.
#       (Hint: Plotting a histogram & distplot helps in recognizing the  discrepencies in prediction, if any.)

# In[139]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.datasets.longley import load_pandas


# In[140]:


gold_data = pd.read_csv("GOLD.csv")
gold_data


# In[141]:


gold_data.set_index('Date',inplace=True)
gold_data


# In[142]:


gold_without_nan = gold_data.dropna()
gold_without_nan


# In[143]:


y = np.array(gold_without_nan["Pred"])
x = np.array(gold_without_nan["new"])
x=x.reshape(-1,1)
y=y.reshape(-1,1)
regression_model = LinearRegression()
regression_model.fit(x,y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y,y_predicted)
r2 = r2_score(y,y_predicted)
print('slop:' ,regression_model.coef_)
print('intercept:',regression_model.intercept_)
print('root mean squared error',rmse)
print('R2 score:',r2)


# In[144]:


plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x,y,s=10)
plt.plot(x,y_predicted,color='r')
plt.show()


# In[145]:


pre_data = gold_data[:]
pre_data_new = pre_data['new']
pre_data_new = pre_data_new.values.reshape(-1,1)


# In[146]:


na_data = regression_model.predict(pre_data_new)
na_data_series = pd.Series(na_data.ravel())
Sata = na_data_series.to_frame()


# In[147]:


#na_data_series.to_frame()
gold_data_frame=pd.DataFrame(gold_data)
gold_data_frame["Pred"] = Sata
gold_data_frame
gold_data.dropna()


# In[148]:


y = np.array(gold_data_frame["Pred"])
x = np.array(gold_data_frame["new"])
x=x.reshape(-1,1)
y=y.reshape(-1,1)
regression_model = LinearRegression()
regression_model.fit(x,y)
y_predicted = regression_model.predict(x)
rmse = mean_squared_error(y,y_predicted)
r2 = r2_score(y,y_predicted)
print('slop:' ,regression_model.coef_)
print('intercept:',regression_model.intercept_)
print('root mean squared error',rmse)
print('R2 score:',r2)


# In[149]:


plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x,y,s=10)
plt.plot(x,y_predicted,color='r')
plt.show()


# In[150]:


plt.hist(gold_data_frame['Pred'])
plt.show()


# In[151]:


sns.distplot(gold_data_frame['Pred'])
plt.show()


# In[152]:


#3.2
tcs_data=pd.read_csv('TCS.csv')
tcs_data['Date'] = pd.to_datetime(tcs_data['Date'])
tcs_data = tcs_data.sort_values('Date')
tcs_data.set_index('Date', inplace=True)
tcs_data


# In[153]:


nifty_data = pd.read_csv('Nifty50.csv')
nifty_data['Date'] = pd.to_datetime(nifty_data['Date'])
nifty_data = nifty_data.sort_values('Date')
nifty_data.set_index('Date', inplace=True)
nifty_data


# In[154]:


file_tcs = tcs_data[405:]
file_nifty = nifty_data[405:]
return_tcs = file_tcs['Close Price'].pct_change()
return_nifty = file_nifty['Close'].pct_change()
plt.figure(figsize=(20,20))
return_tcs.plot()
return_nifty.plot()
plt.ylabel("Daily returns of TCS and NIFTY")
plt.show()


# In[155]:


file_tcs['pct_change'] = file_tcs['Close Price'].pct_change()
file_nifty['pct_change'] = file_nifty['Close'].pct_change()


# In[158]:


y = file_nifty['pct_change'].dropna()
x = file_tcs['pct_change'].dropna()
y = load_pandas().endog
x = load_pandas().exog
x = sm.add_constant(x)
myModel = sm.OLS(y, x).fit()
myModel.summary()


# In[160]:


tcs = pd.read_csv('TCS.csv', parse_dates=True, index_col='Date',)
nifty50 = pd.read_csv('Nifty50.csv', parse_dates=True, index_col='Date')


# In[162]:


monthly_prices = pd.concat([tcs['Close Price'], nifty50['Close']], axis=1)
monthly_prices.columns = ['TCS', 'NIFTY50']


# In[163]:


monthly_prices.head()


# In[164]:


monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0) 
clean_monthly_returns.head()


# In[169]:


X = clean_monthly_returns['TCS']
y = clean_monthly_returns['NIFTY50']
y = load_pandas().endog
x = load_pandas().exog
X1 = sm.add_constant(x)
model=sm.OLS(y,X1)
results = model.fit()
results.summary()


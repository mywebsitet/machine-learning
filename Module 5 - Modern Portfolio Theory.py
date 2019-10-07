
# coding: utf-8

# # Module 5 - Modern Portfolio Theory

#    ### Welcome to the Answer notebook for Module 5 ! 
# Make sure that you've submitted the module 4 notebook and unlocked Module 5 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 5.1 
# 5.1 For your chosen stock, calculate the mean daily return and daily standard deviation of returns, and then just annualise them to get mean expected annual return and volatility of that single stock. **( annual mean = daily mean * 252 , annual stdev = daily stdev * sqrt(252) )**

# In[228]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# In[229]:


itc_data = pd.read_csv('ITC.csv')
itc_data


# In[230]:


itc_data['Daily Return']=(itc_data['Close Price'].pct_change())
itc_data['Daily Return']=itc_data['Daily Return'].replace([np.inf,-np.inf],np.nan)
itc_data = itc_data.dropna()
print("Mean Daily Return")
itc_data['Daily Return'].mean()


# In[231]:


itc_data['Daily Standard Deviation']=(itc_data['Close Price']).pct_change()
itc_data['Daily Standard Deviation']=itc_data['Daily Standard Deviation'].replace([np.inf,-np.inf],np.nan)
itc_data=itc_data.dropna()
print("Daily Standard Deviation")
itc_data['Daily Standard Deviation'].std()


# In[232]:


annual_mean = 0.00015499804624108203 * 252
print("Annual Mean :" + str(annual_mean))


# In[233]:


annual_std = 0.014135221242440118 * math.sqrt(252)
print("Annual Standard Deviation: "+ str(annual_std))


# ### 5.2

# In[234]:


pnb_data = pd.read_csv('PNB.csv')
pnb_data


# In[235]:


ashoka_data = pd.read_csv('ASHOKA.csv')
ashoka_data


# In[236]:


tcs_data = pd.read_csv('TCS.csv')
tcs_data


# In[237]:


itc_data = pd.read_csv('ITC.csv')
itc_data


# In[238]:


idfc_data = pd.read_csv('IDFC.csv')
idfc_data


# In[239]:


data['Pnb'] = pd.DataFrame(pnb_data['Close Price'])
data['Ashoka'] = pd.DataFrame(ashoka_data['Close Price'])
data['Tcs'] = pd.DataFrame(tcs_data['Close Price'])
data['Itc'] = pd.DataFrame(itc_data['Close Price'])
data['Idfc'] = pd.DataFrame(idfc_data['Close Price'])

print("Closing Prics of the 5 respective stock")
data


# ### 5.3

# In[240]:


returns = data.pct_change()
mean_daily_returns=returns.mean()
mean_daily_returns = mean_daily_returns.values.reshape(5,1)
cov_matrix = returns.cov()
weights = np.asarray([0.2,0.2,0.2,0.2,0.2])
portfolio_return = round(np.sum(mean_daily_returns * weigths)*252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))) * np.sqrt(252),2)
print("Portfolio expected annualised return is {} and volatility{}".format(portfolio_return,portfolio_std_dev))


# In[241]:


returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000

results = np.zeros((3,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    weights/=np.sum(weights)
    portfolio_return = np.sum(mean_daily_returns * weights)*252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))) * np.sqrt(252)

    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i]/results[1,i]
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe'])
results_frame


# In[242]:


plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.colorbar()


# In[ ]:


stocks = ['Pnd','Ashoka','Tcs','Itc','Idfc']
returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000

results = np.zeros((4+len(stocks)-1,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    weights/=np.sum(weights)
    portfolio_return = np.sum(mean_daily_returns * weights)*252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))) * np.sqrt(252)

    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i]/results[1,i]
    for j in range(len(weights)):
        results[j+3,i] = weights[j]
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3],stocks[4]])
results_frame


# In[ ]:


max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmax()]

plt.scatter(results_frame.stdev,results_frame.ret,c= results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)


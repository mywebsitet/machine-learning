
# coding: utf-8

# # Module 4 - Algo Trading using Classification
# 

#    ### Welcome to the Answer notebook for Module 4 ! 
# Make sure that you've submitted the module 3 notebook and unlocked Module 4 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 4.1 
# Import the csv file of the stock which contained the Bollinger columns as well.
# 
# 

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as sk
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# In[30]:


tcs_data = pd.read_csv("TCS.csv")
tcs_data['14 Day MA'] = tcs_data['Close Price'].rolling(window=14).mean()
tcs_data['30 Day STD'] = tcs_data['Close Price'].rolling(window=20).std()
tcs_data['Upper Band'] = tcs_data['14 Day MA']+(tcs_data['30 Day STD']*2)
tcs_data['Lower Band'] = tcs_data['14 Day MA']-(tcs_data['30 Day STD']*2)
tcs_data=tcs_data.dropna()
tcs_data['Mid Band'] = (tcs_data['Upper Band']+tcs_data['Lower Band'])/2
tcs_data


# ### Query 4.1a 
# 
# Create a new column 'Call' , whose entries are - 
# 
# >'Buy' if the stock price is below the lower Bollinger band 
# 
# >'Hold Buy/ Liquidate Short' if the stock price is between the lower and middle Bollinger band 
# 
# >'Hold Short/ Liquidate Buy' if the stock price is between the middle and upper Bollinger band 
# 
# >'Short' if the stock price is above the upper Bollinger band
# 
# 
# 

# In[31]:


def select_buy(tcs_data):
    if tcs_data['Close Price'] < tcs_data['Lower Band']:
        return "buy"
    if tcs_data['Close Price'] > tcs_data['Lower Band'] and tcs_data['Close Price'] < tcs_data['Mid Band']:
        return "Hold Buy/ Liquidate Short"
    if tcs_data['Close Price'] > tcs_data['Mid Band'] and tcs_data['Close Price'] < tcs_data['Upper Band']:
        return "Hold Short/ Liquidate Buy"
    if tcs_data['Close Price'] >tcs_data['Upper Band']:
        return "Short"
    
tcs_data = tcs_data.assign(Call = tcs_data.apply(select_buy,axis=1))
tcs_data


# In[32]:


le = preprocessing.LabelEncoder()
train_x = tcs_data[['Upper Band' , 'Lower Band' , 'Mid Band' , 'Close Price']]
trafomed_label = le.fit_transform(tcs_data[['Call']])
train_y = trafomed_label.reshape(-1,1)


# In[33]:


LR = LogisticRegression(random_state=0,solver='lbfgs',multi_class='ovr').fit(train_x,train_y.ravel())
LR.predict(train_x)
print("LOgistic Regression")
round(LR.score(train_x,train_y),4)


# In[34]:


SVM = svm.LinearSVC()
SVM.fit(train_x,train_y)
SVM.predict(train_x)
print("Support Vector Machines")
round(SVM.score(train_x,train_y),4)


# In[35]:


RF = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
RF.fit(train_x,train_y)
RF.predict(train_x)
print("Random Forest")
round(RF.score(train_x,train_y),4)


# In[36]:


NN = MLPClassifier(solver='lbfgs' , alpha = 1e-5 , hidden_layer_sizes=(5,2),random_state=1)
NN.fit(train_x,train_y)
NN.predict(train_x)
print("Neural Networks")
round(NN.score(train_x,train_y),4)


# In[53]:


#4.2
idfc_data = pd.read_csv('IDFC.csv')
idfc_data


# In[54]:


idfc_data['%chg op_cl'] = ((idfc_data['Close Price']-idfc_data['Open Price'])/(idfc_data['Close Price']))*100
idfc_data['%chg lo_hi'] = ((idfc_data['Close Price']-idfc_data['High Price'])/(idfc_data['Low Price']))*100
idfc_data['%chg 5daymean'] = idfc_data['Close Price'].pct_change().dropna().rolling(5).mean()
idfc_data['%chg 5daystd'] = idfc_data['Close Price'].pct_change().dropna().rolling(5).std()
idfc_data


# In[55]:


arr = []
val = []
for value in idfc_data['Close Price'].iteritems():
    arr.append(value[1])
for i in range(0,482):
    if arr[i+1]>arr[i]:
        val.append(1)
    else:
        val.append(-1)
        
idfc_data['Action'] = pd.DataFrame(val)
idfc_data = idfc_data.dropna()
idfc_data


# In[56]:


train_x = idfc_data[['%chg op_cl','%chg lo_hi','%chg 5daymean','%chg 5daystd']]
train_y = idfc_data[['Action']]
RF = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
RF.fit(train_x,train_y)
RF.predict(train_x)
print("Random Forest")
round(RF.score(train_x,train_y),4)


# In[57]:


idfc_data['Net Cummulative Returns'] = (((idfc_data['Open Price']-idfc_data['Close Price'])/(idfc_data['Open Price']))*100).cumsum()
idfc_data


# In[58]:


plt.figure(figsize=(20,20))
plt.plot(idfc_data['Net Cummulative Returns'])


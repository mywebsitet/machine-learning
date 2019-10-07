
# coding: utf-8

# # Module 6 - Clustering for Diverse portfolio analysis

#    ### Welcome to the Answer notebook for Module 6 ! 
# Make sure that you've submitted the module 5 notebook and unlocked Module 6 yourself before you start coding here
# 

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# ### Query 6.1 
# Create a table/dataframe with the closing prices of 30 different stocks, with 10 from each of the caps

# In[4]:


from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
#import pandas_datareader as pdr
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[59]:


gr = pd.DataFrame()
#large
Adaniports = pd.read_csv("ADANIPORTS.csv")
gr['Adaniports'] = Adaniports['Close Price']
Asanpanit = pd.read_csv("ASIANPAINT.csv")
gr['Asanpanit']=Asanpanit['Close Price']
Axisbank = pd.read_csv("AXISBANK.csv")
gr['Axisbank']=Axisbank['Close Price']
Bajfinance = pd.read_csv("BAJFINANCE.csv")
gr['Bajfinance']=Bajfinance['Close Price']
Bpcl = pd.read_csv("BPCL.csv")
gr['Bpcl']=Bpcl['Close Price']
Cipla = pd.read_csv("CIPLA.csv")
gr['Cipla']=Cipla['Close Price']
Drreddy = pd.read_csv("DRREDDY.csv")
gr['Drreddy']=Drreddy['Close Price']
Eichermot = pd.read_csv("EICHERMOT.csv")
gr['Eichermot']=Eichermot['Close Price']
Gail = pd.read_csv("GAIL.csv")
gr['Gail']=Gail['Close Price']
Hdfc = pd.read_csv("HDFC.csv")
gr['Hdfc']=Hdfc['Close Price']

#small
Ashoka = pd.read_csv("ASHOKA.csv")
gr['Ashoka']=Ashoka['Close Price']
Fortis = pd.read_csv("FORTIS.csv")
gr['Fortis']=Fortis['Close Price']
Idfc = pd.read_csv("IDFC.csv")
gr['Idfc']=Idfc['Close Price']
Ircon = pd.read_csv("IRCON.csv")
gr['Ircon']=Ircon['Close Price']
Itdc = pd.read_csv("ITDC.csv")
gr['Itdc']=Itdc['Close Price']
Luxind = pd.read_csv("LUXIND.csv")
gr['Luxind']=Luxind['Close Price']
Ncc = pd.read_csv("NCC.csv")
gr['Ncc']=Ncc['Close Price']
Pvr = pd.read_csv("PVR.csv")
gr['Pvr']=Pvr['Close Price']
Rcom = pd.read_csv("RCOM.csv")
gr['Rcom']=Rcom['Close Price']
Vipind = pd.read_csv("VIPIND.csv")
gr['Vipind']=Vipind['Close Price']





#mid
Dhfl = pd.read_csv("DHFL.csv")
gr['Dhfl']=Dhfl['Close Price']
Mrpl = pd.read_csv("MRPL.csv")
gr['Mrpl']=Mrpl['Close Price']
Suntv = pd.read_csv("SUNTV.csv")
gr['Suntv']=Suntv['Close Price']
Rblbank = pd.read_csv("RBLBANK.csv")
gr['Rblbank']=Rblbank['Close Price']
Tatachem = pd.read_csv("TATACHEM.csv")
gr['Tatachem']=Tatachem['Close Price']
Voltas = pd.read_csv("VOLTAS.csv")
gr['Voltas']=Voltas['Close Price']
Pnb = pd.read_csv("PNB.csv")
gr['Pnb']=Pnb['Close Price']
Idbi = pd.read_csv("IDBI.csv")
gr['Idbi']=Idbi['Close Price']
Igl = pd.read_csv("IGL.csv")
gr['Igl']=Igl['Close Price']
Nbcc = pd.read_csv("NBCC.csv")
gr['Nbcc']=Nbcc['Close Price']
gr['Date']=Adaniports['Date']


# In[60]:


gr.set_index('Date' ,inplace=True)
gr.head(2)


# ### 6.2

# In[64]:


gr = gr.dropna()


# In[65]:


returns = gr.pct_change().mean()*252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = gr.pct_change().std() * sqrt(252)
returns


# ### 6.3

# In[87]:


data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distortions = []
K = range(1,10)
for k in K:
    Kmeanmodel = KMeans(n_clusters = k).fit(X)
    Kmeanmodel.fit(X)
    distortions.append(sum(np.min(cdist(X, Kmeanmodel.cluster_centers_,'euclidean'),axis=1))/X.shape[0])

plt.plot(K,distortions ,'bx-' )
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('The Elbow Method Showing the Optimal K')
plt.show()


# ### 6.4

# In[91]:


centroids,_ = kmeans(X,3)
idx,_ = vq(data,centroids)

plot(data[idx==0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om'
    )
plot(centroids[:,0],centroids[:,1],'sg',markersize=3)
print(returns.idxmax())


# In[94]:


#returns.drop('Piramal' , inplace = True)
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
centroids,_ = kmeans(X,3)
idx,_ = vq(data,centroids)

plot(data[idx==0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om'
    )
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()


# In[97]:


centroids,_ = kmeans(X,3)
idx,_ = vq(data,centroids)

plot(data[idx==0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om'
    )
plot(centroids[:,0],centroids[:,1],'sg',markersize=3)
details = [(name,cluster) for name,cluster in zip(returns.index,idx)]
for detail in details:
    print(detail)


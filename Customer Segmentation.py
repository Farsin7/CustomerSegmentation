#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# # Data Collection and Analysis

# In[4]:


customer_data = pd.read_csv("Mall_Customers.csv")
customer_data.head()


# In[7]:


customer_data.shape


# In[9]:


customer_data.info()


# In[10]:


customer_data.isnull().sum()


# In[11]:


X = customer_data.iloc[:,[3,4]].values
print(X)


# In[12]:


# Choosing the optimal number of clusters
#WCSS

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[13]:


sns.set()
plt.plot(range(1,11), wcss)
plt.title("Elbow Point Graph")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Optimal number of clusters = 5

# # Training K-Means Clustering model

# In[14]:


kmeans = KMeans(n_clusters=5, init="k-means++", random_state=0)

#returning a label for each data point based on their cluster

Y = kmeans.fit_predict(X)

print(Y)


# Clusters = 0,1,2,3,4

# # Visualization of All Clusters

# In[15]:


plt.figure(figsize=(10,10))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='yellow', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='purple', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # MinMax Scaler

# In[1]:


import pandas as pd
df=pd.read_csv('naim.csv')


# In[2]:


df.head()


# In[3]:


x=df.drop('Profit',axis=1)


# In[4]:


x.head()


# In[5]:


y=df[['Profit']]


# In[6]:


y.head()


# In[7]:


from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()


# In[8]:


df_ad=minmax.fit(df[['Administration']])


# In[9]:


df_ad


# In[10]:


x.head()


# In[11]:


df.Administration=minmax.transform(df[['Administration']])


# In[12]:


df.head()


# # MaxAbs Scaler

# In[13]:


import pandas as pd
df=pd.read_csv('naim.csv')


# In[14]:


df.head()


# In[15]:


x=df.drop('Profit',axis=1)


# In[16]:


x.head()


# In[17]:


y=df[['Profit']]


# In[18]:


y.head()


# In[19]:


from sklearn.preprocessing import MaxAbsScaler
maxabs=MaxAbsScaler()


# In[20]:


df_ad=maxabs.fit(df[['Administration']])


# In[21]:


df_ad


# In[22]:


x.head()


# In[23]:


df.Administration=maxabs.transform(df[['Administration']])


# In[24]:


df.head()


# # Robust Scaler

# In[25]:


import pandas as pd
df=pd.read_csv('naim.csv')


# In[26]:


df.head()


# In[27]:


x=df.drop('Profit',axis=1)


# In[28]:


x.head()


# In[29]:


y=df[['Profit']]


# In[30]:


y.head()


# In[31]:


from sklearn.preprocessing import RobustScaler
RoSc=RobustScaler()


# In[32]:


df_ad=RoSc.fit(df[['Administration']])


# In[33]:


df_ad


# In[34]:


x.head()


# In[35]:


df.Administration=RoSc.transform(df[['Administration']])


# In[36]:


df.head()


# In[ ]:





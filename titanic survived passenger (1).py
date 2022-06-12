#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[10]:


titanic_data = pd.read_csv('https://raw.githubusercontent.com/training-ml/Files/main/titanic_train.csv')


# In[11]:


titanic_data.head()


# In[22]:


titanic_data.isnull().sum()


# In[28]:


print(titanic_data['Embarked'].mode())


# In[29]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[ ]:





# In[ ]:





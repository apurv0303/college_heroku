#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics


# In[3]:


df1 = pd.read_csv('Admission_Predict.csv')
df2 = pd.read_csv('Admission_Predict_Ver1.1.csv')


# In[12]:


frames = [df1, df2]
df = pd.concat(frames)
df.tail


# In[5]:


df=df.drop(['Serial No.'], axis = 1)


# In[6]:


X = df.drop(['Chance of Admit '], axis=1).values
y = df['Chance of Admit '].values


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[8]:


rf = RandomForestRegressor(n_estimators = 100, random_state = 0)   
rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)
#print('Root Mean Squared Error for RandomForest:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[9]:


input=[300, 110, 5, 4, 4, 9.5, 1]
final=np.array(input)
final=final.reshape(-1,7)


# In[10]:


My_prediced_chance = rf.predict(final)


# In[11]:


import pickle
pickle.dump(rf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:


#my_chance1=[300, 110, 5, 4, 4, 9.5, 1]


# In[ ]:


#creds=np.array(my_chance1)


# In[ ]:


#my_chance=creds.reshape(-1, 7)


# In[ ]:





# In[ ]:


#My_prediced_chance


# In[ ]:





# In[ ]:





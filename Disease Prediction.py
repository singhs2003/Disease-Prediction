#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('dataset.csv')


# In[3]:


data


# In[4]:


data.count()


# In[5]:


data.isnull().sum()


# In[6]:


data.sample(5)


# In[21]:


#import label encoder
from sklearn import preprocessing 
#make an instance of Label Encoder
le= preprocessing.LabelEncoder()
le.fit(data['Cholesterol'])
le.fit(data['Blood Sugar'])
le.fit(data['Disease'])
le.fit(data['Gender'])
data['Cholesterol'] = le.transform(data['Cholesterol'])
data['Gender'] = le.transform(data['Gender'])
data['Blood Sugar'] = le.transform(data['Blood Sugar'])
data['Disease'] = le.transform(data['Disease'])
data.head()


# In[22]:


min_age=data['Age'].min()
max_age=data['Age'].max()

min_bp=data['Blood Pressure'].min()
max_bp=data['Blood Pressure'].max()


# In[23]:


for i in range(0,4999):
    data['New Age']=(data['Age']-min_age)/(max_age-min_age)
    data['New BP']=(data['Blood Pressure']-min_bp)/(max_bp-min_bp)
    


# In[24]:


data


# In[25]:


data['Blood Pressure']=data['New BP']
data['Age']=data['New Age']


# In[26]:


data


# In[27]:


data=data.drop(['New BP','New Age'],axis=1)


# In[28]:


data


# In[29]:


Y=data['Disease']
X=data.drop('Disease',axis=1)


# In[30]:


X


# In[31]:


Y


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.20, random_state =22)


# In[44]:


X_train


# In[45]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train) 
X_test = sc_x.transform(X_test)


# In[46]:


X_test


# In[47]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)


# In[48]:


Y_pred =model.predict(X_test)


# In[49]:


score = accuracy_score(Y_test,Y_pred)  
print(score)  


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))


# In[157]:


from sklearn.ensemble import RandomForestClassifier


# In[158]:


clf = RandomForestClassifier(n_estimators = 100) 


# In[159]:


clf.fit(X_train, Y_train)


# In[160]:


y_pred = clf.predict(X_test)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# In[167]:


X


# In[168]:


Y


# In[ ]:





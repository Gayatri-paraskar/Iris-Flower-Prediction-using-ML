#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


iris_data = pd.read_csv('Iris.csv')


# In[4]:


iris_data.head()


# In[5]:


iris_data = iris_data.drop(['Id'], axis=1)
iris_data.columns


# In[6]:


print("the dimension:", iris_data.shape)


# In[7]:


print(iris_data.describe())


# In[8]:


print(iris_data.groupby('Species').size())


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


nameplot = iris_data['Species'].value_counts().plot.bar(title='Flower class distribution')
nameplot.set_xlabel('class',size=20)
nameplot.set_ylabel('count',size=20)


# In[11]:


iris_data.plot(kind='box', subplots=True, layout=(2,2), 
               sharex=False, sharey=False, title="Box and Whisker plot for each attribute")
plt.show()


# In[12]:


# plot histogram
iris_data.hist()
plt.show()


# In[13]:


import seaborn as sns
sns.set(style="ticks")
sns.pairplot(iris_data, hue="Species")


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


# we will split data to 80% training data and 20% testing data with random seed of 10
X = iris_data.drop(['Species'], axis=1)
Y = iris_data['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


# In[17]:


print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_train.shape:", X_train.shape)
print("Y_test.shape:", Y_test.shape)


# In[26]:



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


# In[36]:


# models
models = []
models.append(('KNN', KNeighborsClassifier()))
names.append(name)
pd_results = cross_val_score(model, X_train, Y_train, scoring='accuracy')
accuracy.append(pd_results)
msg = "%s: accuracy=%f std=(%f)" % (name, pd_results.mean(), pd_results.std())
print(msg)


# In[37]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[45]:


# models
models = []
models.append(('KNN', KNeighborsClassifier()))


# In[46]:


def test_model(model):
    model.fit(X_train, Y_train) # train the whole training set
    predictions = model.predict(X_test) # predict on test set
    
    # output model testing results
    print("Accuracy:", accuracy_score(Y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print("Classification Report:")
    print(classification_report(Y_test, predictions))


# In[42]:


# predict values with our test set
for name, model in models:
    print("Y_test")
    print("Testing", name)
    test_model(model)


# ### Project made by : Gayatri Paraskar.

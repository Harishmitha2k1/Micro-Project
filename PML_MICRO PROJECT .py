#!/usr/bin/env python
# coding: utf-8

#    # PML MICRO PROJECT
#  
# ### CLASSIFICATION OF  ANIMALS
#  
#  
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# In[2]:


#Load the data
zoo = pd.read_csv('zoo.csv')
class_df = pd.read_csv('class.csv')


# In[3]:


zoo.head()


# In[4]:


zoo['class_type'].unique()


# In[5]:


class_df.head(7)


# In[6]:


class_df['Class_Type'].unique()


# In[7]:


zoo.columns


# In[8]:


class_df.columns


# In[9]:


zoo.dtypes


# In[10]:


class_df.dtypes


# In[11]:


zoo.info()


# In[12]:


class_df.info()


# In[13]:


zoo.describe()


# In[14]:


class_df.describe()


# In[15]:


#Combine the zoo and the class files
zoo = zoo.merge(class_df,how='left',left_on='class_type',right_on='Class_Number')


# ### Brief Exploratory Analysis

# In[16]:


zoo.head()


# In[17]:


print(zoo.columns)

print(zoo['animal_name'].unique() )

print(zoo['Class_Type'].unique() )


# ### Visualization of data

# In[18]:


plt.figure(figsize=(10,10))
sns.light_palette("seagreen", as_cmap=True)
fig = sns.countplot(x=zoo['Class_Type'],label="Count", palette = "Greens_r")
fig = fig.get_figure()


# In[19]:


#run a correlation plot on all our feautres and see if any arise.

plt.figure(figsize=(13,13))
corr = zoo.iloc[:,1:-1].corr()
sns.heatmap(corr, cmap = "Greens", annot=True)
plt.show()


# ### Test Splitting

# In[20]:


# Remove unwanted columns, and assign the x and y values
features = ['hair', 'feathers', 'eggs', 'milk', 'airborne',
       'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
       'fins', 'legs', 'tail', 'domestic', 'catsize']

X = zoo[features]
y = zoo['Class_Number']


# In[21]:


X_train,X_test,y_train,y_text = train_test_split(X,y , test_size = 0.25, random_state = 42)


# ### K-Nearest Neighbors

# In[22]:


#Implement KNN
X = zoo[features]
y = zoo['Class_Number']
X_train,X_test,y_train,y_text = train_test_split(X,y , test_size = 0.25, random_state = 42)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_train)
print("Accuracy: {}".format(accuracy_score(y_train, y_pred)))


# ## Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

X = zoo[features]
y = zoo['Class_Number']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=42)

lor = LogisticRegression(penalty='l2', C=10.0)
lor.fit(X_train, y_train)
y_pred = lor.predict(X_test)

accuracy_score(y_test,y_pred)


# ## svm

# In[24]:


X = zoo[features]
y = zoo['Class_Number']

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Declare the model
svm = SVC(kernel='linear', C=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=42)

# Train the model
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

#Get Accuracy Score
accuracy_score(y_pred_svm,y_test)


# ## Decision tree

# In[25]:


dec = DecisionTreeClassifier(random_state=1)
dec = dec.fit(X_train,y_train)
pred = dec.predict(X_test)

plt.figure(figsize=(25,25))
df= tree.plot_tree(dec,feature_names = features,filled = True)

print("Accuracy: " + str(accuracy_score(y_test,y_pred)))


# ###  Logistic Regression is selected as a best model

# In[26]:


from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

X  = zoo[features]
y = zoo['Class_Number']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=42)

lor = LogisticRegression(penalty='l2', C=10.0)
lor.fit(X_train, y_train)
y_pred = lor.predict(X_test)


# In[27]:


from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix


# In[28]:


print('accuracy_score:', accuracy_score(y_pred, y_test))

confusion_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(data = confusion_matrix)


# In[29]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[30]:


print("Recall:", recall_score(y_test, y_pred, average='micro'))


# In[31]:


print("Precision:", precision_score(y_test, y_pred, average='micro'))


# In[ ]:





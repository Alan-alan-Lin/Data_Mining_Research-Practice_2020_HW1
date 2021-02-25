#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data = pd.read_csv("character-deaths.csv")
data.info()


# In[2]:


#把空值以0替代
data.isna().sum()


# In[3]:


select_data = pd.DataFrame(data)
select_data['Death Year'] = select_data['Death Year'].fillna(0)
select_data['Book of Death'] = select_data['Book of Death'].fillna(0)
select_data['Death Chapter'] = select_data['Death Chapter'].fillna(0)
select_data['Book Intro Chapter'] = select_data['Book Intro Chapter'].fillna(0)
select_data


# In[4]:


# Death Year , Book of Death , Death Chapter三者取一個，將有數值的轉成1 
select_data['Death Year'] = np.where(select_data['Death Year'] > 0, 1.0, 0.0)
select_data


# In[5]:


#將Allegiances轉成dummy特徵(底下有幾種分類就會變成幾個特徵，值是0或1，本來的資料集就會再增加約20種特徵)
select_data_2 = pd.get_dummies(select_data['Allegiances'])
select_data_2


# In[6]:


select_data_3 = select_data.join(select_data_2)
select_data_3.info()


# In[7]:


#亂數拆成訓練集(75%)與測試集(25%) 
from sklearn.model_selection import train_test_split
X = select_data_3.iloc[:,5:34]
y = select_data_3.loc[:,"Death Year"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
X_test


# In[8]:


#使用scikit-learn的DecisionTreeClassifier進行預測
#criterion : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different 
#            attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.
#max_depth : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. 
#            The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).
from sklearn import tree
from sklearn import metrics
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
game_clf = clf.fit(X_train, y_train)
predicted = game_clf.predict(X_test)
accuracy = metrics.accuracy_score(predicted, y_test)
print(accuracy)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))


# In[10]:


#產出決策樹的圖(限制樹的深度=3) 
fn = ["Book of Death","Death Chapter","Book Intro Chapter","Gender","Nobility","GoT","CoK","SoS","FfC","DwD","Arryn","Baratheon","Greyjoy","House Arryn","House Baratheon","House Greyjoy","House Lannister","House Martell","House Stark","House Targaryen","House Tully","House Tyrell","Lannister","Martell","Night's Watch","Stark","Targaryen","Tully","Tyrell","Wildling"]
cn = ["0","1"]
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=500)

tree.plot_tree(game_clf,feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[ ]:





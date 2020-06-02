#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv('/data/training/tic-tac-toe.data.txt')

#le = LabelEncoder()
train.iloc[:,0] = LabelEncoder().fit_transform(train.iloc[:,0])
train.iloc[:,1] = LabelEncoder().fit_transform(train.iloc[:,1])
train.iloc[:,2] = LabelEncoder().fit_transform(train.iloc[:,2])
train.iloc[:,3] = LabelEncoder().fit_transform(train.iloc[:,3])
train.iloc[:,4] = LabelEncoder().fit_transform(train.iloc[:,4])
train.iloc[:,5] = LabelEncoder().fit_transform(train.iloc[:,5])
train.iloc[:,6] = LabelEncoder().fit_transform(train.iloc[:,6])
train.iloc[:,7] = LabelEncoder().fit_transform(train.iloc[:,7])
train.iloc[:,8] = LabelEncoder().fit_transform(train.iloc[:,8])
train.iloc[:,9] = LabelEncoder().fit_transform(train.iloc[:,9])

X = train.iloc[:, 0:9]
y = train.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

re = RandomForestClassifier(n_estimators = 100)

kfold = KFold(n_splits=20, random_state=0)
cross_val = cross_val_score(re, X_train, y_train, scoring='accuracy', cv=kfold)
print(np.round(cross_val.mean(), 3))

ad = AdaBoostClassifier(base_estimator=re, n_estimators=100, learning_rate=1, random_state=0)
ad.fit(X_train, y_train)

y_pred = ad.predict(X_test)
print(np.round(accuracy_score(y_test, y_pred), 3))

result=[np.round(cross_val.mean(), 3), np.round(accuracy_score(y_test, y_pred), 3)]

result=pd.DataFrame(result)

#writing output to output.csv
result.to_csv('/code/output/output.csv', header=False, index=False)


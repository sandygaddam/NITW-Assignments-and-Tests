#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score

train=pd.read_csv('/data/training/Pacific_train.csv')
test=pd.read_csv('/data/test/Pacific_test.csv')

train['Status'].value_counts()
test['Status'].value_counts()

features = ['Maximum Wind', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW','High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']
label = ['Status']

sns.heatmap(train[features].corr())

features_selected = ['Maximum Wind', 'Minimum Pressure', 'Low Wind SE']

X_train = train[features_selected]
y_train = train[label]

from sklearn.tree import DecisionTreeClassifier
clf_DC = DecisionTreeClassifier()
clf_DC.fit(X_train, y_train)
cvs_DC = cross_val_score(clf_DC, X_train, y_train, cv=10)
#print(cvs_DC)
cvs_DC.mean()

from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators=200)
clf_RF.fit(X_train, y_train)
cvs_RF = cross_val_score(clf_RF, X_train, y_train, cv=10)
#print(cvs_RF)
cvs_RF.mean()

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)
cvs_gnb = cross_val_score(clf_gnb, X_train, y_train, cv=10)
#print(cvs_gnb)
cvs_gnb.mean()

from sklearn.svm import SVC
clf_svc = SVC(gamma='auto')
clf_svc.fit(X_train, y_train)
cvs_svc = cross_val_score(clf_svc, X_train, y_train, cv=10)
#print(cvs_svc)
cvs_svc.mean()

X_test = test[features_selected]
y_test = test[label]

pred_DC = clf_DC.predict(X_test)
pred_RF = clf_RF.predict(X_test)
pred_gnb = clf_gnb.predict(X_test)
pred_SVC = clf_svc.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_matrix_DC = confusion_matrix(y_test, pred_DC)
conf_matrix_RF = confusion_matrix(y_test, pred_RF)
conf_matrix_gnb = confusion_matrix(y_test, pred_gnb)
conf_matrix_SVC = confusion_matrix(y_test, pred_SVC)

def precision(label, confusion_matrix):
    pre = confusion_matrix[:, label]
    return confusion_matrix[label, label] / pre.sum()

def recall(label, confusion_matrix):
    rec = confusion_matrix[label, :]
    return confusion_matrix[label, label] / rec.sum()

def accuracy(confusion_matrix):
    diag_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diag_sum / sum_of_all_elements

all_models = [conf_matrix_DC, conf_matrix_RF, conf_matrix_gnb, conf_matrix_SVC]
algorithms = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'SupportVectorClassifier']

for each in range(len(algorithms)):
    print('For each in : ', algorithms[each])
    print('Accuracy :', np.round(accuracy(all_models[each]), 2))
    for label in range(10):
        print(f'{label:5d} {precision(label, all_models[each]):9.3f} {recall(label, all_models[each]):6.3f}')
        print()

result = ['RandomForestClassifier', 0.96]
result=pd.DataFrame(result)
#writing output to output.csv
result.to_csv('/code/output/output.csv', header=False, index=False)


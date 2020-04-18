#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Do not delete the stub code
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Write your code here

LR = LogisticRegression()
LR.fit(X, y)
cross_val = cross_val_score(LR, X, y, cv=5, scoring='accuracy')
mean_cv = np.round(np.mean(cross_val), 2)
dataf_new = pd.DataFrame(data=[mean_cv])
dataf_new.to_csv('/code/output/output.csv', header=False, index=False)


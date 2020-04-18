#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries here
# import numpy as np
# from sklearn import linear_model
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

# User can save any number of plots to `/code/output` directory
# Note that only the plots saved in /code/output will be visible to you
# So make sure you save each plot as shown below
# Uncomment the below 3 lines and give it a try
# axes = pd.tools.plotting.scatter_matrix(iris_train, alpha=0.2)
# plt.tight_layout()
# plt.savefig('/code/output/scatter_matrix.png')

cars = pd.read_csv('/data/training/mtcars.csv')

X = cars.drop(['model', 'mpg'], axis=1)
y = cars['mpg']

lasso = Lasso()
ridge = Ridge()

lasso.fit(X,y)
ridge.fit(X,y)

cross_val_lasso = cross_val_score(lasso, X, y, cv=10)
cross_val_ridge = cross_val_score(ridge, X, y, cv=10)

mean_value = np.round((np.mean(cross_val_lasso) - np.mean(cross_val_ridge)), 2)
#print('mean_value : ', mean_value)

dataf = pd.DataFrame([mean_value])
dataf.to_csv('/code/output/output.csv', header=None, index=None)


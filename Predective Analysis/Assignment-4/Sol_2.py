#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""
I.Load 1000 rows from the dataset from the file bottle.csv.
"""

dataf_all = pd.read_csv("bottle.csv", nrows=1000)
dataf_ST = dataf_all[['Salnty','T_degC']]

dataf_ST['T_degC'] = dataf_ST['T_degC'].fillna(dataf_ST['T_degC'].mean())
dataf_ST['Salnty'] = dataf_ST['Salnty'].fillna(dataf_ST['Salnty'].mean())

X = dataf_ST['Salnty']
y = dataf_ST['T_degC']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
#print('x_train',x_train.shape, '\ny_train',y_train.shape, '\nx_test',x_test.shape, '\ny_test',y_test.shape)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

LR = LinearRegression()
LR.fit(x_train, y_train)
#print('Model Coeff :', LR.coef_, '\nModel Interc :', LR.intercept_)

intercept = [np.round(LR.intercept_, 3)]
coefficient = [np.round(LR.coef_, 3).tolist()]
#print(intercept, coefficient)

y_pred = LR.predict(x_test)

MSE = np.round(mean_squared_error(y_test, y_pred), 3)
R2_score = np.round(r2_score(y_test, y_pred), 3)
#print('Mean Squared Error : ', MSE)
#print('R2 Score :', R2_score)

num_folds = 5
kf = KFold(n_splits=5, random_state=5, shuffle=True)
score = cross_val_score(LR, np.array(X).reshape(-1,1), y, cv=kf, scoring='neg_mean_squared_error')

convert_score = 0 - score 

RMSE = [np.sqrt(np.abs(a)) for a in convert_score]

RMSE_mean = np.round(np.mean(RMSE), 3)

dataf_1 = pd.DataFrame([x_train.shape[0], x_test.shape[0], coefficient, intercept])
dataf_1.to_csv('output1.csv', header=None, index=None)

dataf_2 = pd.DataFrame([MSE, R2_score, RMSE_mean])
dataf_2.to_csv('output2.csv', header=None, index=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





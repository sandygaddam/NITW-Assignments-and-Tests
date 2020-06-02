#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math

forecast=pd.read_csv('/data/training/PredictionsFor4April2019.csv')

de = forecast[(forecast['Country_code'] == 'DE')]
at = forecast[(forecast['Country_code'] == 'AT')]
pl = forecast[(forecast['Country_code'] == 'PL')]

de = de[['ActualValue', 'PredValue']]
at = at[['ActualValue', 'PredValue']]
pl = pl[['ActualValue', 'PredValue']]

de_rmse = np.round(math.sqrt(mean_squared_error(de['ActualValue'], de['PredValue'])), 2)
at_rmse = np.round(math.sqrt(mean_squared_error(at['ActualValue'], at['PredValue'])), 2)
pl_rmse = np.round(math.sqrt(mean_squared_error(pl['ActualValue'], pl['PredValue'])), 2)


result=[de_rmse, at_rmse, pl_rmse]
result=pd.DataFrame(result)
#writing output to output.csv
result.to_csv('/code/output/output.csv', header=False, index=False)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix

forecast=pd.read_csv('/data/training/PredictionsFor4April2019.csv')

de = forecast[(forecast['Country_code'] == 'DE')]
at = forecast[(forecast['Country_code'] == 'AT')]
pl = forecast[(forecast['Country_code'] == 'PL')]

de = de[['ActualValue', 'PredValue']]
at = at[['ActualValue', 'PredValue']]
pl = pl[['ActualValue', 'PredValue']]

de_cm = confusion_matrix(de['ActualValue'], de['PredValue'])
at_cm = confusion_matrix(at['ActualValue'], at['PredValue'])
pl_cm = confusion_matrix(pl['ActualValue'], pl['PredValue'])

de_per = np.round(((de_cm[0][0]+de_cm[1][1])/len(de['ActualValue']))*100, 3)
at_per = np.round(((at_cm[0][0]+at_cm[1][1])/len(at['ActualValue']))*100, 3)
pl_per = np.round(((pl_cm[0][0]+pl_cm[1][1])/len(pl['ActualValue']))*100, 3)

result=[de_per, at_per, pl_per]
result=pd.DataFrame(result)
#writing output to output.csv
result.to_csv('/code/output/output.csv', header=False, index=False)


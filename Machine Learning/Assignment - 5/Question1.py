#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os

#reading from file Titanic_train.csv
dataf = pd.read_csv('Titanic_train.csv')

dataf1 = dataf[dataf['Survived'] == 1]
dataf2 = dataf[dataf['Survived'] == 0]

result = [len(dataf1[dataf1['Pclass']==3]), len(dataf2[dataf2['Sex']=='male']), len(dataf1[dataf1['Embarked']=='S'])]

result = pd.DataFrame(result)

#writing output to output.csv
result.to_csv('output.csv', header=False, index=False)


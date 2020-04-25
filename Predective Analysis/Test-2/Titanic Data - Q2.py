#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv

dataf = pd.read_csv('/data/training/titanic.csv')

with open('/code/output/output1.csv', 'w', newline='') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow([str(np.round(dataf['Age'].mean(), 3))])
    writer.writerow([str(np.round(dataf['Age'].std(), 3))])
    writer.writerow([str(np.round(dataf['SibSp'].mean(), 3))])
    writer.writerow([str(np.round(dataf['SibSp'].std(), 3))])
    writer.writerow([str(np.round(dataf['Parch'].mean(), 3))])
    writer.writerow([str(np.round(dataf['Parch'].std(), 3))])
    writer.writerow([str(np.round(dataf['Fare'].mean(), 3))])
    writer.writerow([str(np.round(dataf['Fare'].std(), 3))])
    
with open('/code/output/output2.csv', 'w', newline='') as out1:
    writer = csv.writer(out1, delimiter=',')
    per = len(dataf[dataf['Survived'] == 1])/len(dataf['Survived'])
    writer.writerow([str(np.round(per*100, 3))])
    only_female = dataf[dataf['Sex'] == 'female']
    prob_female = len(only_female)/len(dataf)
    only_male = dataf[dataf['Sex'] == 'male']
    prob_male = len(only_male)/len(dataf)
    all_par_age = dataf[dataf['Survived']==1]
    prob_par_male = len(all_par_age[all_par_age['Sex'] == 'male'])/len(only_male)
    prob_par_female = len(all_par_age[all_par_age['Sex'] == 'female'])/len(only_female)
    writer.writerow([str(np.round(prob_par_female, 3))])
    writer.writerow([str(np.round(prob_par_male, 3))])
    dataf1 = dataf[dataf['Pclass'] == 1]
    dataf2 = dataf[dataf['Pclass'] == 3]
    dataf1_mean = np.round(dataf1['Age'].mean(), 3)
    dataf2_mean = np.round(dataf2['Age'].mean(), 3)
    writer.writerow([str(dataf1_mean - dataf2_mean)])


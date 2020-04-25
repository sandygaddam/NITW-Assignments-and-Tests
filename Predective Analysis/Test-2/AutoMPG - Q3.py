#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
from scipy.stats import mode 
from scipy.stats import pearsonr 

dataf = pd.read_csv('/data/training/autompg.csv')

mpg_mean = dataf['mpg'].mean()
mpg_median = int(dataf['mpg'].median())
mpg_mode = mode(dataf['mpg'])
mpg_std = dataf['mpg'].std()

a = pearsonr(dataf['weight'], dataf['mpg'])
b = dataf['weight'].corr(dataf['mpg'], method = 'kendall')

with open('/code/output/output.csv', 'w', newline='') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow([str(np.round(mpg_mean, 2))])
    writer.writerow([str(np.round(mpg_median))])
    writer.writerow([str(np.round(mpg_mode[0][0]))])
    writer.writerow([str(np.round(mpg_std, 2))])
    writer.writerow([str(np.round(b, 2))])
    writer.writerow([str(np.round(a[0], 2))])

file = open('/data/training/testcaseauto.txt')
testcase = file.readline().strip()

for i in range(1, int(testcase)+1):
    sam1 = file.readline().strip()
    dataf1 = pd.read_csv('/data/training/{}.csv'.format(sam1))
    sam1_mean = dataf1['mpg'].mean()
    sam1_std = dataf1['mpg'].std()
    with open('/code/output/output{}.csv'.format(i), 'w', newline='') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow([str(np.round((mpg_mean - sam1_mean), 2))])
        writer.writerow([str(np.round((mpg_std - sam1_std), 2))])


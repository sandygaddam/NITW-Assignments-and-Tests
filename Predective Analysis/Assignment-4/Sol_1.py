#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import csv
from scipy import stats

# I. Load data
iq_data=np.loadtxt("iqdata.csv")
iq1 = iq_data[0:10000]

f = open("testcaseiq.txt")
nooftestcases = f.readline().strip()

for i in range(1, int(nooftestcases)+1):
    with open("output{}.csv".format(i), "w") as out:
        writer = csv.writer(out, delimiter=",")
        writer = csv.writer(out)
        writer.writerow([str(round(np.mean(iq1),2))])
        writer.writerow([str(round(np.std(iq1),2))])
        lower_value = f.readline().strip()
        upper_value = f.readline().strip()
        probability = np.subtract(stats.norm(np.mean(iq1),np.std(iq1)).cdf(int(upper_value)), stats.norm(np.mean(iq1),np.std(iq1)).cdf(int(lower_value)))*100
        writer.writerow([str(np.round(probability,3))])
        file = f.readline().strip()
        sample = pd.read_csv("{}.csv".format(file))
        p_value = stats.ttest_1samp(a=sample,popmean=np.mean(iq1))
        if p_value[1][0] < 0.05:
            writer.writerow(["Reject"])
        else:
            writer.writerow(["Accept"])


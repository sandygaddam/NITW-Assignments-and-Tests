import os
import pandas as pd
import numpy as np
from scipy import stats

os.chdir(r'E:\Edureka\Predictive Analytics\Assignment-3')

def main():

    dataf = pd.read_csv('blackfriday.csv')
    
    var_purchase = round(np.var(dataf['Purchase']), 2)
    sd_purchase = round(np.std((dataf['Purchase'])), 2)
    skew_purchase = round(stats.skew(dataf['Purchase']), 2)
    kurt_purchase = round(stats.kurtosis(dataf['Purchase']), 2)
    
    dataf_new = pd.DataFrame(data=[var_purchase, sd_purchase, skew_purchase, kurt_purchase])
    dataf_new.to_csv('output.csv', index=False, header=False)

main()
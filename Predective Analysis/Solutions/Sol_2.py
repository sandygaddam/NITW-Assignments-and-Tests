import os
import numpy as np
import pandas as pd

os.chdir(r'E:\Edureka\Predictive Analytics\Assignment-3')


def main():
    dataf = pd.read_csv('blackfriday.csv')
    
    mean_purchase = round(np.mean(dataf['Purchase']), 2)
    median_purchase = round(np.median(dataf['Purchase']), 1)
    mode_purchase = list(dataf['Purchase'].mode())
    
    df_new = pd.DataFrame(data=[mean_purchase, median_purchase])
    df_new = df_new.append(mode_purchase)

main()

import os
import pandas as pd
from scipy.stats import iqr

os.chdir(r'E:\Edureka\Predictive Analytics\Assignment-3')

def main():

    dataf = pd.read_csv('blackfriday.csv')
    count = 0
    
    i_qr = round(iqr(dataf['Purchase']), 1)
    q_1 = dataf['Purchase'].quantile(0.25)
    q_3 = dataf['Purchase'].quantile(0.75)
    
    for value in dataf['Purchase']:
        if value < (q_1 - 1.5 * i_qr) or value > (q_3 + 1.5 * i_qr):
            count += 1
    
    dataf_new = pd.DataFrame([i_qr, count])
    dataf_new.to_csv('output.csv', header=False, index=False)

main()
import os
import pandas as pd

os.chdir(r'E:\Edureka\Predictive Analytics\Assignment-3')

def main():
    
    dataf = pd.read_csv('blackfriday.csv')
    dataf_1 = dataf.isna().sum()
    dict_1 = dataf_1.to_dict()
    keys = []
    add = 0
    
    for key, value in dict_1.items():
        if value > 0:
            keys.append(key)
            add = add + 1
    
    dataf_new = pd.DataFrame(data=[add])
    dataf_new = df_new.append(keys)
    dataf_new.to_csv('output.csv', index=False, header=False)

main()
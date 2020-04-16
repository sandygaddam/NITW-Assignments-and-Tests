import os
import pandas as pd

os.chdir(r'E:\Edureka\Predictive Analytics\Assignment-3')

def main():
    
    file = open('testcaseprobability.txt', 'r')
    age_list = []
    category_list = []
    count = 0
    
    for line in file:
        if count != 0:
            if line[len(line)-1] == '\n':
                if count%2 != 0:
                    age_list.append(line[:len(line)-1])
                else:
                    category_list.append(line[:len(line)-1])
            else:
                if count%2 != 0:
                    age_list.append(line)
                else:
                    category_list.append(line)
        count += 1                
    
    dataf = pd.read_csv('blackfriday.csv')
    
    for i in range(0, len(age_list)):
        dataf_1 = dataf[dataf['Age'] == age_list[i]]
        counts = dataf_1['Gender'].value_counts()
        gender = counts.index[0]
        gender_map = {'M':'Male', 'F':'Female'}
        gender = gender_map[gender]
        print(f'{gender} {age_list[i]}')
        category_prob = round(len(dataf[dataf['City_Category'] == category_list[i]])/len(dataf), 4)
        print(f'{category_prob} {category_list[i]}')
        dataf_new = pd.DataFrame(data=[gender, category_prob])
        dataf_new.to_csv(f'output{i+1}.csv', header=False, index=False)

main()
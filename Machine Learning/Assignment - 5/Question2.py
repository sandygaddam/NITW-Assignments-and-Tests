# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 04:48:02 2020

@author: Home
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
os.chdir(r'C:\Users\******\Desktop\folder_new\training-Titanic_train')
dataset = pd.read_csv('Titanic_train.csv')
test_ds = pd.read_csv('test.csv')
na_valcount = dataset['Age'].isnull().sum()
indexes =  dataset[dataset['Age'].isnull()].index.tolist()
sum_indexes = sum(indexes)
dataset['Embarked'] = dataset['Embarked'].fillna('S')
X_df = dataset[~dataset['Age'].isnull()][['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_df = dataset[~dataset['Age'].isnull()]['Age'].values
y_df = y_df.reshape(-1,1)

enc_Sex = LabelEncoder()
enc_Embarked = LabelEncoder()
X_df['Sex'] = enc_Sex.fit_transform(X_df['Sex'])
X_df['Embarked'] = enc_Embarked.fit_transform(X_df['Embarked'])
scale_X = StandardScaler()
scale_y = StandardScaler()
X_df = scale_X.fit_transform(X_df)
y_df = scale_y.fit_transform(y_df)

model=LinearRegression()
model.fit(X_df,y_df)
to_predds = dataset[dataset['Age'].isnull()][['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
to_predds['Sex'] = enc_Sex.transform(to_predds['Sex'])
to_predds['Embarked'] = enc_Embarked.transform(to_predds['Embarked'])
to_predds = scale_X.transform(to_predds)

y_pred_scaled = model.predict(to_predds)
y_pred = scale_y.inverse_transform(y_pred_scaled)
mean_ypred = round(y_pred.mean(), 2)

list_out = [na_valcount, sum_indexes, mean_ypred]
out = pd.DataFrame(list_out)

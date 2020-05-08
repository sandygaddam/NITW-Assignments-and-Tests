import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import confusion_matrix as cm
os.chdir(r'C:\Users\******\Desktop\folder_new\training-Titanic_train')
train = pd.read_csv('Titanic_train.csv')
test= pd.read_csv('test.csv')
#preparing index
train_null_index =  train[train['Age'].isnull()].index.tolist()
test_null_index = test[test['Age'].isnull()].index.tolist()
#cleaning and splitting for Linear Regression
train['Embarked'] = train['Embarked'].fillna('S')
trainX_notna= train[~train['Age'].isnull()][['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
trainy_notna= train[~train['Age'].isnull()][['Age']].values
trainy_notna = trainy_notna.reshape(-1,1)

test_na= test[test['Age'].isnull()][['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
train_na = train[train['Age'].isnull()][['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']]
#training dataset encoded and scaled

enc_Sex = LabelEncoder()
enc_Embarked = LabelEncoder()
scale_X = StandardScaler()
scale_y = StandardScaler()
trainX_notna['Sex'] = enc_Sex.fit_transform(trainX_notna['Sex'])    
trainX_notna['Embarked'] = enc_Embarked.fit_transform(trainX_notna['Embarked'])
trainX_notna = scale_X.fit_transform(trainX_notna)
trainy_notna = scale_y.fit_transform(trainy_notna)

#predicting dataset encoded and scaled

train_na['Sex'] = enc_Sex.transform(train_na['Sex'])
train_na['Embarked'] = enc_Embarked.transform(train_na['Embarked'])
test_na['Sex'] = enc_Sex.transform(test_na['Sex'])
test_na['Embarked'] = enc_Embarked.transform(test_na['Embarked'])
test_na = scale_X.transform(test_na)
train_na = scale_X.transform(train_na)

#building linear regression model for predicting age na values
model = LinearRegression()
model = model.fit(trainX_notna,trainy_notna)
na_vals_fortrain = list(scale_y.inverse_transform(model.predict(train_na)))
na_vals_fortest = list(scale_y.inverse_transform(model.predict(test_na)))

#intergrating predicted age values into the traindata
for i ,j in zip(train_null_index,na_vals_fortrain):
    train.at[i, 'Age'] = j[0]
for i ,j in zip(test_null_index,na_vals_fortest):
    test.at[i, 'Age'] = j[0]
#finding the count of adults and children
child_count = len(train[(train['Age'] < 18) & (train['Survived']==1)])
adult_count = len(train[(train['Age'] >= 18)& (train['Survived']==1)])
total_count = child_count + adult_count
# preparing data for Logistic regression for predicting survival

train_X = train.drop(['Unnamed: 0','PassengerId','Name','Ticket','Cabin','Survived'],axis=1)
train_y = train['Survived']
test_X = test.drop(['Unnamed: 0','PassengerId','Name','Ticket','Cabin','Survived'],axis=1)
test_y = test['Survived']
#encoding and scaling the X part
train_X['Sex'] = enc_Sex.fit_transform(train_X['Sex'])
train_X['Embarked'] = enc_Embarked.fit_transform(train_X['Embarked'])
test_X['Sex'] = enc_Sex.transform(test_X['Sex'])
test_X['Embarked'] = enc_Embarked.transform(test_X['Embarked'])

train_X = scale_X.fit_transform(train_X)
test_X = scale_X.transform(test_X)
#building the Log reg model

classifier = LogisticRegression()
classifier.fit(train_X,train_y)
#predicting values
y_predicted = classifier.predict(test_X)
# confusion matrix
mat = cm(test_y,y_predicted)
matsum = mat[0][0]+mat[0][1]+mat[1][0]+mat[1][1]
accuracy = ((mat[0][0]+mat[1][1])/(matsum))*100

#writing csv
result = [total_count ,matsum ,accuracy]
out_df = pd.DataFrame(result)
out_df.to_csv('output.csv', header=False, index=False)

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

import os

# 1) Reading test and train data

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 2) Cleaning data 

# a) Removing columns that logically don't/should not be co-related to survival

train_data = train_data.drop(['Name','PassengerId','Ticket','Cabin'], axis=1)
test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)

# b) Combining teat and train data

data_set = [train_data, test_data]

# 3) Preprocessing data

# a) Mapping to nominal categorical variables

sex_mapping = {'male' : 0, 'female' : 1}
data_set = [data.replace({'Sex': sex_mapping}) for data in data_set]

# b) 

for data in data_set:
    for i in range(0,2):
        for j in range(0,3):
            data.loc[np.isnan(data['Age']) & (data['Sex'] == i) & (data['Pclass'] == j+1), 'Age'] = np.median(data[(data['Sex']) == i & (data['Pclass'] == j+1)]['Age'].dropna())

# c) Turning age to bins

for data in data_set:
    data['AgeBand'] = pd.cut(data['Age'], 5)

for data in data_set:
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4
    data['Age'] = data['Age'].astype(int)
    
for data in data_set:
    data.drop(['AgeBand'], axis=1, inplace = True)

# d) Creating family size as a new feature and removing features unnecessary features

for data in data_set:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

# e) Mapping to nominal categorical variables
		# -> first finding mode of embarked variable so as to set that for those rows where 'embarked' is na

embark_mode = data_set[0].Embarked.dropna().mode()[0]
embarked_mapping = {'S': 0,'C': 1,'Q': 2}

for data in data_set:
    data['Embarked'].fillna(embark_mode, inplace = True)
    data.replace({'Embarked': embarked_mapping}, inplace = True)

# f) Filling fares that are na with median fares and then cutting it into bins

data_set[1]['Fare'].fillna(data_set[1]['Fare'].dropna().median(), inplace = True)

data_set[0]['FareBand'] = pd.qcut(data_set[0]['Fare'], 4)

for data in data_set:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

data_set[0].drop(['FareBand'], axis = 1, inplace = True)

passengerID = data_set[1]["PassengerId"]

X_train = data_set[0].drop("Survived", axis = 1)
Y_train = data_set[0]["Survived"]
X_test  = data_set[1].drop("PassengerId", axis = 1)

# 4) Applying ML techniques - STACKING used

# a) Creating different ml models to serve as primary models

rf = RandomForestClassifier()
svc = SVC()
nb = GaussianNB()
knn = KNeighborsClassifier()
percep = Perceptron()

# b) Creating test and train data

X_train_first_level = X_train[:round(0.8*len(X_train))]
X_test_first_level = X_train[-round(0.2*len(X_train)):]
Y_train_first_level = Y_train[:round(0.8*len(Y_train))]
Y_test_first_level =Y_train[-round(0.2*len(Y_train)):]

# c) Training models

rf.fit(X_train_first_level, Y_train_first_level)
svc.fit(X_train_first_level, Y_train_first_level)
nb.fit(X_train_first_level, Y_train_first_level)
knn.fit(X_train_first_level, Y_train_first_level)
percep.fit(X_train_first_level, Y_train_first_level)

# d) Obtaining results of the models

results__ = [rf.predict(X_test_first_level), svc.predict(X_test_first_level), nb.predict(X_test_first_level), knn.predict(X_test_first_level), percep.predict(X_test_first_level)]

results__df = pd.DataFrame(data=results__).transpose()

# e) Using predictions to train a higher level learner

rf_final = RandomForestClassifier()
rf_final.fit(results__df,  Y_test_first_level)

results__ = [rf.predict(X_test), svc.predict(X_test), nb.predict(X_test), knn.predict(X_test), percep.predict(X_test)]
results__df = pd.DataFrame(data=results__).transpose()

# f) Storing results

result = pd.DataFrame({'PassengerId':passengerID, 'Survived':rf_final.predict(results__df)})

result.to_csv(path_or_buf = "submission.csv", index = False)

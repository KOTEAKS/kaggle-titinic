import pandas as pd
import numpy as np
import csv as csv
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train_df = pd.read_csv('train.csv', header=0)

# Columns
# id            PassengerId: int e.g. 1, 2, 3, ...
# survival      Survival
#               (0 = No; 1 = Yes)
# pclass        Passenger Class
#               (1 = 1st; 2 = 2nd; 3 = 3rd)
# name          Name
# sex           Sex
# age           Age
# sibsp         Number of Siblings/Spouses Aboard
# parch         Number of Parents/Children Aboard
# ticket        Ticket Number
# fare          Passenger Fare
# cabin         Cabin
# embarked      Port of Embarkation
#               (C = Cherbourg; Q = Queenstown; S = Southampton)

# sex to gender
train_df['Gender_F'] = np.where(train_df['Sex'] == 'female', 1, 0)
train_df['Gender_M'] = np.where(train_df['Sex'] == 'male', 1, 0)

# embarked
train_df.Embarked[ train_df.Embarked.isnull() ] = 'NA'

Ports = list(enumerate(np.unique(train_df['Embarked'])))

for i, name in Ports:
    train_df['Embarked_' + name] = np.where(train_df['Embarked'] == name, 1, 0)

# name
train_df['SirNames'] = train_df['Name'].map(lambda x: x.split(".")[1].split("(")[0].strip())
train_df['Titles'] = train_df['Name'].map(lambda x: x.split(",")[1].split(".")[0].split(" ")[1].strip())

# family
#train_df['Family'] = train_df['Parch'] + train_df['SibSp'] + 1
#train_df['Alone'] = train_df['Family'].map(lambda x: 1 if x == 1 else 0)
#train_df['Small'] = train_df['Family'].map(lambda x: 1 if 2 <= x <= 4 else 0)
#train_df['Large'] = train_df['Family'].map(lambda x: 1 if x > 4 else 0)

#SirNames = list(enumerate(np.unique(train_df['SirNames'])))
Titles = list(enumerate(np.unique(train_df['Titles'])))
#print train_df[train_df.SirNames == '']
for i, name in Titles:
    train_df['Titles_' + name] = np.where(train_df['Titles'] == name, 1, 0)

# age
median_age = train_df['Age'].dropna().median()
median_age_of_survived = train_df.Age[train_df.Survived == 1].dropna().median()
median_age_of_not_survived = train_df.Age[train_df.Survived == 0].dropna().median()

train_df.loc[ (train_df.Age.isnull()) & (train_df.Survived == 1), 'Age'] = median_age_of_survived
train_df.loc[ (train_df.Age.isnull()) & (train_df.Survived == 0), 'Age'] = median_age_of_not_survived

# drop columns
#train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Embarked_NA', 'SirNames'], axis=1) 
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Embarked_NA', 'SirNames',
'PassengerId', 'Titles'], axis=1) 

# TEST DATA
test_df = pd.read_csv('test.csv', header=0)

# save ids
ids = test_df['PassengerId'].values

# sex to gender
test_df['Gender_F'] = np.where(test_df['Sex'] == 'female', 1, 0)
test_df['Gender_M'] = np.where(test_df['Sex'] == 'male', 1, 0)

# embarked
test_df.Embarked[ test_df.Embarked.isnull() ] = 'NA'
for i, name in Ports:
    test_df['Embarked_' + name] = np.where(test_df['Embarked'] == name, 1, 0)

# names
test_df['Titles'] = test_df['Name'].map(lambda x: x.split(",")[1].split(".")[0].split(" ")[1].strip())
for i, name in Titles:
    test_df['Titles_' + name] = np.where(test_df['Titles'] == name, 1, 0)

# family
#test_df['Family'] = test_df['Parch'] + test_df['SibSp'] + 1
#test_df['Alone'] = test_df['Family'].map(lambda x: 1 if x == 1 else 0)
#test_df['Small'] = test_df['Family'].map(lambda x: 1 if 2 <= x <= 4 else 0)
#test_df['Large'] = test_df['Family'].map(lambda x: 1 if x > 4 else 0)

#median_age = test_df['Age'].dropna().median()
test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# pclass to fare
median_fare = np.zeros(3)
for f in range(0, 3):
    median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()

for f in range(0, 3):
    test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# passenger id
#test_df.PassengerId = test_df.PassengerId - train_df.PassengerId.max()

# drop columns
#test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Embarked_NA'], axis=1)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Embarked_NA',
'PassengerId', 'Titles'], axis=1)

import math

for column in list(train_df.columns):
    c_max = train_df[column].max()
    if c_max > 0:
        train_df[column] = train_df[column] / train_df[column].max()

for column in list(test_df.columns):
    c_max = test_df[column].max()
    if c_max > 0:
        test_df[column] = test_df[column] / c_max

train_data = train_df.values
test_data = test_df.values

classifiers = [
        KNeighborsClassifier(10),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=10),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        RandomForestClassifier(max_depth=10, n_estimators=100),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

counts = dict()
for id in ids:
    counts[id] = 0

for forest in classifiers:
    print 'Training... ' + forest.__class__.__name__
    forest = forest.fit(train_data[0::,1::], train_data[0::,0])

    print 'Predicting...' + forest.__class__.__name__
    output = forest.predict(test_data).astype(int)

    for id, result in zip(ids, output):
        counts[id] += result

n_classfiers = len(classifiers)
output = [1 if result > n_classfiers / 2 else 0 for id, result in counts.iteritems()]

predictions_file = open("koteaks.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

print 'Done.'

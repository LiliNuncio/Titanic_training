# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:26:57 2017

@author: ax28957
"""

# Data analysis and wrangling
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

#load data
train= pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))

train.info()
train.head()

#number of survived
train['Survived'].value_counts(normalize=True)
sns.countplot(train['Survived'])

#survuved by class
train['Survived'].groupby(train['Pclass']).mean()
sns.countplot(train['Pclass'], hue=train['Survived'])

#Split by Name Title
#lambda anonymous function 
train['Name'].head()
train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Name_Title'].value_counts()


#Grouping by Name Title
#pd.qcut(x, q, labels=None, retbins=False, precision=3) - quantile based on discretization function
train['Survived'].groupby(train['Name_Title']).mean()

#Long Names
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()

pd.qcut(train['Name_Len'],5).value_counts()

#divided by Sex -  total passengers
train['Sex'].value_counts(normalize=True)

#divided by sex survived
train['Survived'].groupby(train['Sex']).mean()


#survived by age, verify the % that has this data
train['Survived'].groupby(train['Age'].isnull()).mean()

#survived by age
train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()

pd.qcut(train['Age'],5).value_counts()

#sibssp
train['Survived'].groupby(train['SibSp']).mean()
train['SibSp'].value_counts()

#ticket
train['Ticket'].head(n=10)
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
train.groupby(['Ticket_Len'])['Survived'].mean()
train['Ticket_Len'].value_counts()

#ticket first letter
train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
train['Ticket_Lett'].value_counts()
train.groupby(['Ticket_Lett'])['Survived'].mean()

#Fare ticket
pd.qcut(train['Fare'], 3).value_counts()
train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean()

#Relation between Class and fare
pd.crosstab(pd.qcut(train['Fare'], 5), columns=train['Pclass'])

#cabin leter
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train['Cabin_Letter'].value_counts()
train['Survived'].groupby(train['Cabin_Letter']).mean()

#cabin number
train['Cabin_num'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
train['Cabin_num'].replace('an', np.NaN, inplace = True)
train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x <> '' else np.NaN)

pd.qcut(train['Cabin_num'],3).value_counts()
train['Survived'].groupby(pd.qcut(train['Cabin_num'], 3)).mean()

#correlation between survived and cabin number
train['Survived'].groupby(pd.qcut(train['Cabin_num'], 3)).mean()

#embarked
train['Embarked'].value_counts()
train['Embarked'].value_counts(normalize=True)
train['Survived'].groupby(train['Embarked']).mean()
sns.countplot(train['Embarked'], hue=train['Pclass'])

# >> Modelling
#Creation of two columns: Lenght name and title name
def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test

#null values of age filling by average of pessengers by title and class
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    train['mean'] = train.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')
    train['Age'] = train['Age'].fillna(train['mean'])
    z = test.merge(train, on=['Name_Title', 'Pclass'], how='left').drop_duplicates(['PassengerId_x'])
    test['Age'] = np.where(test['Age'].isnull(), z['mean'], test['Age'])
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    del train['mean']
    return train, test

#combine the SibSp and Parch columns into a new variable that indicates family size, and group the family size variable into three categories.
def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test

#ticket length
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


#extraction of first letter of cabin and it number

def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x <> '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test

#fill the embarket data with the most common data : 'S'
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test

#fill the fare data with the average
test['Fare'].fillna(train['Fare'].mean(), inplace = True)


#convertion of our categorical columns into dummyvariables.

def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

#drops any columns that haven't already been dropped
def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test

# execute function in order to build a dataset
train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 
                                              'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = drop(train, test)

#check columns on dataset
len(train.columns)


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1, 5, 10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])


print(gs.best_score_)
print(gs.best_params_)
#print(gs.cv_results_)

# model estimation and evaluation
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print "%.4f" % rf.oob_score_

#variable importance
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]

#predict the target variable 
np.shape(test)


predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv(os.path.join('data', 'test.csv'))
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv(os.path.join('submission_files', 'y_test.csv'), sep=",", index = False)





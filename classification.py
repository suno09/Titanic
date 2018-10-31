import itertools
import re
import sys
import pickle
from time import time, strftime, gmtime
from copy import deepcopy

import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

start_time = time()

# * Data preprocessing
# read csv file train and test
dataframe = pd.read_csv('data/train.csv')

# Create column NameType(Mr, Mrs, ...) from Name
dataframe['NameType'] = [
    re.sub(r'.+?, (.+?)\..*', r'\1', name) for name in dataframe.Name
]
dataframe.NameType = dataframe.NameType.replace(
    ['Sir', 'Capt', 'Major', 'Don', 'Rev', 'Jonkheer', 'Col'],
    "Mr"
)
dataframe.NameType = dataframe.NameType.replace(
    ['Mlle', 'Lady', 'Mme', 'Miss', 'Mrs', 'the Countess', 'Dona'],
    "Ms"
)

# replace row contains sex = female and nameType = Dr by Ms
dataframe.loc[
    (dataframe.Sex == "female") & (dataframe.NameType == "Dr"), 'NameType'
] = "Ms"
# replace row contains sex = male and nameType = Dr by Mr
dataframe.loc[
    (dataframe.Sex == "male") & (dataframe.NameType == "Dr"), 'NameType'
] = "Mr"

# create family size
dataframe['FamilySize'] = dataframe.SibSp + dataframe.Parch
# Fare per passenger (fare per ticket)
dataframe['FarePerPerson'] = dataframe.Fare / (dataframe.FamilySize + 1)

# replace the NaN with "Unknown"
dataframe.Cabin.fillna('', inplace=True)
# create the count of Cabin
dataframe['CabinCount'] = dataframe.Cabin.apply(
    lambda c: len(str(c).strip().split())
)
# create the number of Cabin (nbr after the letter)
# (if exist one or more of cabins so get the first else 0)
dataframe['NbrOfCabin'] = dataframe.Cabin.apply(
    lambda cs: list(filter(
        lambda c: len(c) > 1, str(cs).strip().split()
    ))
)
dataframe.NbrOfCabin = [
    int(c[0][1:]) if c != [] else 0 for c in dataframe.NbrOfCabin
]
# Create Column and take the first letter of the cabin and
dataframe['CabinType'] = dataframe.Cabin
dataframe.CabinType = [c[0] if c else 'U' for c in dataframe.CabinType]
# person per Cabin
dataframe['PersonPerCabin'] = (dataframe.FamilySize + 1) / dataframe.CabinCount
dataframe.loc[
    dataframe.PersonPerCabin == np.inf, 'PersonPerCabin'
] = dataframe.loc[dataframe.PersonPerCabin == np.inf, 'FamilySize']
# Fare of cabin
dataframe['FareOfCabin'] = dataframe.Fare / dataframe.CabinCount
dataframe.loc[
    (dataframe.FareOfCabin == np.inf) | dataframe.FareOfCabin.isnull(),
    'FareOfCabin'
] = 0
# # Create column and set the cabin by position
#  A = best pos, ..., G = worst pos, U = worst pos
# cabin_pos = {v: 1./k for k, v in enumerate(
# ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'U'], 1)}
# dataframe['CabinPos'] = [cabin_pos[c] for c in dataframe.CabinType]

#
dataframe.Age = dataframe.groupby(["NameType"]).transform(
    lambda a: a.fillna(a.mean())
).Age
#
dataframe.Embarked.fillna('U', inplace=True)

# the new dataframe with deleting some columns
df_prep = dataframe.drop(
    ['PassengerId', 'Name', 'Ticket', 'Cabin'],
    axis=1
)

# get the list of target Survived
y = df_prep.Survived.values
df_prep = df_prep.drop(['Survived'], axis=1)

# get the cols of type "object"
# object_cols = ['Sex', 'Embarked', 'NameType', 'CabinType']
object_cols = df_prep.select_dtypes("object").columns
# Convert the categorical data to numbers
label_encoders = {}
for obj_col in object_cols:
    label_encoder = LabelEncoder()
    label_encoder.fit(df_prep[obj_col])
    df_prep[[obj_col]] = df_prep[[obj_col]].apply(label_encoder.transform)
    label_encoders[obj_col] = label_encoder

# generate the classifiers
# key : name of algorithm
# value : list of size 2 which contains algorithm and params of algo
classifiers = {
    "Logistic Regression": [LogisticRegression, {'random_state': 0}],
    "KNN": [KNeighborsClassifier,
            {'n_neighbors': 1, 'metric': 'minkowski', 'p': 2}],
    "SVM rbf": [SVC, {'kernel': 'rbf', 'random_state': 0}],
    "SVM poly": [SVC, {'kernel': 'poly', 'random_state': 0}],
    "SVM sigmoid": [SVC, {'kernel': 'sigmoid', 'random_state': 0}],
    "SVM precomputed": [SVC, {'kernel': 'precomputed', 'random_state': 0}],
    "SVM linear": [SVC, {'kernel': 'linear', 'random_state': 0}],
    "Naive Bayes": [GaussianNB, {}],
    "Decision Tree": [DecisionTreeClassifier, {'criterion': "entropy",
                                               'random_state': 0}],
    "Random Forest": [RandomForestClassifier,
                      {'n_estimators': 10,
                       'criterion': 'entropy',
                       'random_state': 0}]
}
len_algorithms = classifiers.__len__()
# count of all columns in data
len_cols = len(df_prep.columns)
end_nbr_f = df_prep.columns.__len__()
# end_nbr_f = 3
start_nbr_f = end_nbr_f // 5 + 1
# start_nbr_f = start_nbr_f if start_nbr_f != 0 else 1
nbr_tests = sum(
    len(list(itertools.combinations(range(len_cols), nbr_features))) for
    nbr_features in range(start_nbr_f, end_nbr_f + 1)
) * len_algorithms

# use K fold
kf = KFold(n_splits=9)

# classifiers = {
#     "Logistic Regression": LogisticRegression(**{'random_state': 0}),
#     "KNN": KNeighborsClassifier(
#         **{'n_neighbors': 5, 'metric': 'minkowski', 'p': 2}),
#     "SVM": SVC(**{'kernel': 'rbf', 'random_state': 0}),
#     "Naive Bayes": GaussianNB(),
#     "Decision Tree": DecisionTreeClassifier(**{'criterion': "entropy",
#                                                'random_state': 0})
#     ,
#     "Random Forest": RandomForestClassifier(**
#                                             {'n_estimators': 10,
#                                              'criterion': 'entropy',
#                                              'random_state': 0})
# }

results = []

# generate classifiers
for nbr_features in range(start_nbr_f, end_nbr_f + 1):
    for indexes_cols in itertools.combinations(range(len_cols), nbr_features):
        columns = [col for index, col in enumerate(df_prep.columns)
                   if index in indexes_cols]
        x = df_prep[columns].values.astype(np.float64)

        # get indexes of type objects
        indexes_object_cols = [index for index, col in enumerate(columns)
                               if col in object_cols]

        # use dummy variables for columns of types object
        if indexes_object_cols:
            onehotencoder = OneHotEncoder(
                categorical_features=indexes_object_cols
            )
            x = onehotencoder.fit_transform(x).toarray()
            # x_test = onehotencoder.transform(x_test).toarray()
        else:
            onehotencoder = None

        # Feature Scaling
        sc = StandardScaler()
        x = sc.fit_transform(x)
        # x_test = sc.transform(x_test)

        for name_classifier, [classifier, params] in classifiers.items():
            # extract columns
            # Start generate model
            # print(name_classifier, columns)
            ml_classifier = classifier(**params)
            scores = cross_val_score(ml_classifier, x, y, cv=kf)
            # print(name_classifier, " => ", scores)
            # ml_classifier.fit(x_train, y_train)

            # save the result
            # print(name_classifier)
            results.append({
                "x_y": [x, y],
                "dummy_var": onehotencoder,
                "feature_scaling": sc,
                "classifier": ml_classifier,
                "features": list(columns),
                "algorithm": name_classifier,
                "accuracy": max(scores)
            })

            # print the progression
            progress_value = results.__len__() * 100. / nbr_tests
            sys.stdout.write("\r")
            sys.stdout.write("Progression |%-50s| %.2f %% (%s)" %
                             ("\u2588" * int(progress_value / 2.),
                              progress_value,
                              strftime("%H:%M:%S", gmtime(time() - start_time))
                              )
                             )
            sys.stdout.flush()

# sort and view results
max_accuracy = max([d['accuracy'] for d in results])
best_classifier = min(
    filter(lambda d: d['accuracy'] == max_accuracy, results),
    key=lambda d: len(d['features'])
)

end_time = time()

print("\nThe best classifier is %s" % best_classifier['algorithm'])
print("The accuracy : %.2f %%" % (max_accuracy * 100.))
print("The columns : %s" % best_classifier['features'])
print("The duration of execution : %s" %
      strftime("%H hours %M minutes %S seconds", gmtime(end_time - start_time))
      )
print("Number of tests = %d tests" % results.__len__())

# add the test data to best classifier
x, y = best_classifier['x_y']
best_classifier['classifier'].fit(x, y)

# save label encoders
best_classifier['label_encoders'] = label_encoders

# Save best classifier to pickle as dictionary
with open('best_classifier.pik', 'wb') as wp:
    pickle.dump(best_classifier, wp)

import itertools
import re
import sys
from queue import Queue
from threading import Thread, Lock
from time import time, strftime, gmtime

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


start_time = time()

print_lock = Lock()

index_th = 0
results = []


def thread_classifier(ml_class, str_classifier, cols,
                      x_tr, x_te, y_tr, y_te,
                      q: Queue, tnbr_tests):

    # Start generate model
    ml_class.fit(x_tr, y_tr)

    # save the result
    q.put({
        "features": cols,
        "algorithm": str_classifier,
        "accuracy": accuracy_score(y_te, ml_class.predict(x_te))
    })

    # print the progression
    global index_th
    index_th += 1
    progress_value = index_th * 100. / tnbr_tests
    with print_lock:
        sys.stdout.write("\r")
        sys.stdout.write("Progression |%-100s| %.2f %%" %
                         ("\u2588" * int(progress_value), progress_value)
                         )
        sys.stdout.flush()


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
    ['Mlle', 'Lady', 'Mme', 'Miss', 'Mrs', 'the Countess'],
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
y_target = df_prep.Survived.values
df_prep = df_prep.drop(['Survived'], axis=1)

# get the cols of type "object"
# object_cols = ['Sex', 'Embarked', 'NameType', 'CabinType']
object_cols = df_prep.select_dtypes("object").columns
# Convert the categorical data to numbers
df_prep[object_cols] = df_prep[object_cols].apply(LabelEncoder().fit_transform)

# generate the classifiers
# key : name of algorithm
# value : list of size 2 which contains algorithm and params of algo
classifiers = {
    "Logistic Regression": [LogisticRegression, {'random_state': 0}],
    "KNN": [KNeighborsClassifier,
            {'n_neighbors': 5, 'metric': 'minkowski', 'p': 2}],
    "SVM rbf": [SVC, {'kernel': 'rbf', 'random_state': 0}],
    # "SVM poly": [SVC, {'kernel': 'poly', 'random_state': 0}],
    # "SVM sigmoid": [SVC, {'kernel': 'sigmoid', 'random_state': 0}],
    # "SVM precomputed": [SVC, {'kernel': 'precomputed', 'random_state': 0}],
    # "SVM linear": [SVC, {'kernel': 'linear', 'random_state': 0}],
    "Naive Bayes": [GaussianNB, {}],
    "Decision Tree": [DecisionTreeClassifier, {'criterion': "entropy",
                                               'random_state': 0}]
    ,
    "Random Forest": [RandomForestClassifier,
                      {'n_estimators': 10,
                       'criterion': 'entropy',
                       'random_state': 0}]
}
# generate threads for multiple ML algorithms
len_algorithms = classifiers.__len__()
# count of all columns in data
len_cols = len(df_prep.columns)
start_nbr_f = 1
end_nbr_f = df_prep.columns.__len__()
# end_nbr_f = 3
nbr_tests = sum(
    len(list(itertools.combinations(range(len_cols), nbr_features))) for
    nbr_features in range(start_nbr_f, end_nbr_f + 1)
) * len_algorithms

# generate the threads of classifiers
threads = []
queue = Queue()
for nbr_features in range(start_nbr_f, end_nbr_f + 1):
    for indexes_cols in itertools.combinations(range(len_cols), nbr_features):
        columns = [col for index, col in enumerate(df_prep.columns)
                   if index in indexes_cols]
        # generate the input and target of train and test
        x = df_prep[columns].values.astype(np.float64)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_target,
            test_size=0.25,
            random_state=0
        )

        # get indexes of type objects
        indexes_object_cols = [
            index for index, col in enumerate(columns) if col in object_cols
        ]

        # use dummy variables for columns of types object
        if indexes_object_cols:
            onehotencoder = OneHotEncoder(
                categorical_features=indexes_object_cols
            )
            x_train = onehotencoder.fit_transform(x_train).toarray()
            x_test = onehotencoder.transform(x_test).toarray()

        # Feature Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        for name_classifier, [classifier, params] in classifiers.items():
            # extract columns
            # Start generate model
            # print(name_classifier)
            threads.append(Thread(
                target=thread_classifier,
                args=(
                    classifier(**params), name_classifier, columns,
                    x_train, x_test, y_train, y_test,
                    queue, nbr_tests
                )
            ))

# start threads and wait all results
_ = [thread.start() for thread in threads]
_ = [thread.join() for thread in threads]
results = [queue.get() for thread in threads]

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

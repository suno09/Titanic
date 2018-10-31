import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the classifier
with open('classifiers/best_classifier_pca.pik', 'rb') as rp:
    classifier = pickle.load(rp)

with open('classifiers/dataframe_titanic.pik', 'rb') as rp:
    dataframe = pickle.load(rp)
df_prep = dataframe[891:]

object_cols = df_prep.select_dtypes("object").columns

# Convert the categorical data to numbers
label_encoders = classifier['label_encoders']
for obj_col in object_cols:
    label_encoder = LabelEncoder()
    label_encoder.fit(df_prep[obj_col])
    df_prep[[obj_col]] = df_prep[[obj_col]].apply(label_encoder.transform)
    label_encoders[obj_col] = label_encoder

# generate the input of test
x = df_prep.drop(['Survived'], 1).values.astype(np.float64)

# transform data test with PCA Algotirhm
x = classifier['pca'].transform(x)

# predict with classifier
print("Predict with %s" % classifier['algorithm'])

y = classifier['classifier'].predict(x)
df_test = pd.DataFrame(
    data=list(zip(range(892, 892 + len(df_prep)), y.astype(np.int))),
    columns=['PassengerId', 'Survived']
)

df_test.to_csv(path_or_buf='classifiers/predict_test_titanic.csv', sep=',',
               index=False)

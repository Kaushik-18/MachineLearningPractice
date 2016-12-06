import matplotlib.pyplot as pyplot
from matplotlib import style
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

style.use("ggplot")

df = pd.read_excel('data-sets/titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def convert_non_numeric_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_val = {}

        def convert_to_int(vals):
            return text_digit_val[vals]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = convert_non_numeric_data(df)
print(df.head())
df.drop(['boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predictee = np.array(X[i].astype(float))
    predictee = predictee.reshape(-1, len(predictee))
    prediction = clf.predict(predictee)
    if prediction[0] == Y[i]:
        correct += 1

print(correct / len(X))

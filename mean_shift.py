import matplotlib.pyplot as pyplot
from matplotlib import style
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import numpy as np

style.use("ggplot")

df = pd.read_excel('data-sets/titanic.xls')
orignal_df = pd.DataFrame.copy(df)
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
clusters = clf.cluster_centers_

orignal_df['cluster_group'] = np.nan
for i in range(len(X)):
    orignal_df['cluster_group'].iloc[i] = labels[i]

num_clusters = len(np.unique(labels))
survival_rate = {}
for i in range(num_clusters):
    temp_df = orignal_df[(orignal_df['cluster_group'] == float(i))]
    survive_cluster = temp_df[(temp_df['survived'] == 1)]
    survival = len(survive_cluster)/len(temp_df)
    survival_rate[i] = survival

print(survival_rate)








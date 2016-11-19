import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd

df = pd.read_csv('data-sets/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

# accuracy = clf.score(x_test, y_test)
# print(accuracy)

prediction_data = np.array([4,2,1,1,1,2,3,2,1]) #,[4,2,1,2,2,2,3,2,1])
prediction_data = prediction_data.reshape(1,-1)
prediction = clf.predict(prediction_data)
print(prediction)

import pandas as pn
import quandl, math, datetime
import numpy as np

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Note : below data set cannot actually be used to predict the stock price.. This is just a example to demonstrate L.R
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_CHANGE']]

forecast_col = 'Adj. Close'
df.fillna('-999', inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_forecast = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

print(x_forecast)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
# documentation says we can run on multiple threads
clf = LinearRegression()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)

forecast_set = clf.predict(x_forecast)
# print(forecast_set,score,forecast_out)

# below steps are required plotting ,we basically take the last available date and create rows in data frame so that
# predictions can be plotted
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
df['Forecast'] = np.nan

for i in forecast_set:
    print(i)
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

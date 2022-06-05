import math
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('./dataset/tesla_stocks.csv')

# print(dataframe.head())
# print(dataframe.info())
# print(dataframe.describe())
# print(dataframe.isnull().sum())
# print(dataframe.columns)
# print(dataframe.shape)
# print(dataframe["Volume"].unique())

dataframe['Volatile'] = dataframe.apply(lambda row: math.fabs(row['Open'] - row['Close']) > 0.15 * row['Open'], axis=1)

X = dataframe[['Open', 'Volume']]
y = dataframe['Volatile']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = SVC(kernel='linear')

model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)
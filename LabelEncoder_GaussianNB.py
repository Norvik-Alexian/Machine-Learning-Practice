import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('./dataset/iris.csv')

# necessary investigations
print(data.columns)
print(data.shape)
print(data.head())
print(data.tail())
print(data.dtypes)
print(data.describe())
print(data.isnull().sum())
print(data['variety'].unique())

X = data[['sepal.length', 'petal.width']]
y = data['variety']

# making the labels numeric
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
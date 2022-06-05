import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

data = pd.read_csv('./dataset/iris.csv')

print(data.head())
print(data.columns)
print(data.shape)
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

model = LogisticRegression()
scores = cross_val_score(model, X, y)

print(scores)

avg_score = scores.mean()

print(avg_score)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

scores = cross_val_score(model, X, y, cv=kfold)

print(scores)
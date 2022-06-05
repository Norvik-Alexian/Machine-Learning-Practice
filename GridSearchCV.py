import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./dataset/iris.csv')

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

print(pd.array(y).unique())

d_tree = DecisionTreeClassifier()
param_grid = {'max_depth': [4, 3, 5, 6], 'criterion': ['gini', 'entropy']}

g_search = GridSearchCV(d_tree, param_grid=param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

g_search.fit(X_train, y_train)

score = g_search.score(X_test, y_test)
estimator = g_search.best_estimator_
params = g_search.best_params_

print(params)
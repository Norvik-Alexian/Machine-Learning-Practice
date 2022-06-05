import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def polynomial_formula(x, deg, coef):
    result = 0
    for i in range(deg):
        result += coef[i] * x ** i

    return result


x_vector = np.random.randint(1, 10, 5)
coef = np.random.randint(1, 3, 5)
deg = len(coef)

poly_result = [polynomial_formula(x, deg, coef) for x in x_vector]

plt.plot(x_vector, poly_result)
plt.show()


df = pd.read_csv('dataset/titanic.csv')

print(df.head())
print(df.columns)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df['Survived'].unique())

# data preprocessing
df['Survived'] = df['Survived'].astype(np.float64)
df['Pclass'] = df['Pclass'].astype(np.float64)

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Sex'] = df['Sex'].astype(np.float64)
df['Age'] = df['Age'].fillna(df.Age.median())

X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

model = SVC(kernel='poly', degree=5)
# model = SVC(kernel='rbf')
# model = SVC(kernel='sigmoid')
# model = SVC(kernel='linear')

model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(accuracy)
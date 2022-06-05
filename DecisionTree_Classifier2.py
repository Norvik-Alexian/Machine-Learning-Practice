import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('./dataset/tic-tac-toe.csv')

for col in data.columns:
    data[col], _ = pd.factorize(data[col], sort=True)


X = data.drop('positive', axis=1)
y = data['positive']

sns.countplot(y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

for i in range(5, 20):
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print(i, accuracy)

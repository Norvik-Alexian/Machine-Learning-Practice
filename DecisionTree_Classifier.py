import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('./dataset/diabetes.csv')

print(data.head())
print(data.dtypes)
print(data.describe())
print(data.columns)
print(data['Outcome'].unique())

X = data[['Glucose', 'Insulin']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

model = DecisionTreeClassifier(max_depth=2, criterion='gini')

model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

plot_tree(
    model,
    feature_names=['Glucose', 'Insulin'],
    class_names=['Absent', 'Present'],
    filled=True
)

plt.show()

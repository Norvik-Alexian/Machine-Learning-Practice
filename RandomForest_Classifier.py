import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('./dataset/spam.csv', sep='\t')

print(data.dtypes)
print(data.describe())
print(data.isnull().sum())
print(data.columns)
print(data['Type'].unique())

label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])

count_vectorizer = CountVectorizer()
count_vectorizer.fit(data['Message'].values)

X = count_vectorizer.transform(data['Message']).toarray()
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

rand_forest_model = RandomForestClassifier(n_estimators=20)

rand_forest_model.fit(X_train, y_train)

prediction = rand_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print(accuracy)

decision_tree_model = DecisionTreeClassifier(max_depth=15)
decision_tree_model.fit(X_train, y_train)

prediction2 = decision_tree_model.predict(X_test)
accuracy2 = accuracy_score(y_test, prediction2)

print(accuracy2)

feature_importance = pd.Series(rand_forest_model.feature_importances_).sort_values(ascending=False)

print(feature_importance)
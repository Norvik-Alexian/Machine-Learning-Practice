import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

wine_dataframe = pd.read_csv('./dataset/wine.csv')

wine_dataframe['high alcohol'] = wine_dataframe['Alcohol'].apply(lambda x: 1 if x > 13.4 else 0)

X = wine_dataframe[['Malic.acid', 'Mg']]
y = wine_dataframe['high alcohol']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

model = LogisticRegression()

model.fit(X_train, y_train)  # train the model
prediction = model.predict(X_test)  # prediction of the model

accuracy = accuracy_score(y_test, prediction)  # find the accuracy of the model

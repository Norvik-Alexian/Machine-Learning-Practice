import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

dataframe = pd.read_csv('./dataset/titanic.csv')

print(dataframe.info())
print(dataframe.columns)
print(dataframe.isnull().sum())

X = dataframe[['Pclass', 'SibSp', 'Survived']]

model = KMeans(2)
model.fit(X)

prediction = list(model.predict(X))

plt.plot(dataframe['Survived'], 'r^')
plt.plot(prediction, 'b.')
plt.xlim((100, 150))

plt.show()

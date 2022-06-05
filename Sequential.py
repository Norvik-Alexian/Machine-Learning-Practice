import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('./dataset/wine.csv')

print(data.head())
print(data.columns)
print(data.dtypes)
print(data.describe())
print(data.isnull().sum().sum())
print(data.shape)

data['high_alc'] = data['Alcohol'].apply(lambda x: 1 if x > 3.14 else 0)

X = data[['Malic.acid', 'Mg']]
y = data['high_alc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = Sequential()

model.add(Dense(12, activation='relu', input_shape=(2, )))
model.add(Dense(8, activation='relu', input_shape=(2, )))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=20, verbose='auto')

score = model.evaluate(X_test, y_test)
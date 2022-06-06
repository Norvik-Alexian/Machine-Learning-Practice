import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns


df = pd.read_csv('./dataset/titanic.csv')

# print(df.head())

# print(df.info())
# print(df.describe())
print(df.isnull().sum())

# print(df['sex'].unique())
# print(df['cabin'].unique().shape)
# print(df['ticket'].unique().shape)
# print(df['embarked'].unique())

# print(df.columns)

# print(df['age'].describe())
# print(df['age'].head(10))

df = df.dropna(subset=['embarked', 'fare'])

df['age'].fillna(df['age'].median(), inplace=True)

df['age'].hist()
plt.show()
exit()
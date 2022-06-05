import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./dataset/iris.csv')

print(data.head())
print(data.isnull().sum())
print(data.columns)
print(data.dtypes)
print(data.describe())

# sns.countplot(data['Close'])

plt.matshow(data.corr())
plt.xticks(range(data.shape[1]), data.columns, fontsize=12, rotation=90)
plt.yticks(range(data.shape[1]), data.columns, fontsize=12)

color_bar = plt.colorbar()
color_bar.ax.tick_params(labelsize=12)

encoder = LabelEncoder()

data['variety'] = encoder.fit_transform(data['variety'])

print(data['variety'].unique())

data.hist()
plt.show()

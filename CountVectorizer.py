import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv('./dataset/train.csv').drop(columns=['id'])
test_df = pd.read_csv('./dataset/test.csv').drop(columns=['id'])

df = pd.concat((train_df, test_df))

df.dropna(inplace=True)

count_model_title = CountVectorizer()
count_model_text = CountVectorizer()

count_vec_title = count_model_title.fit_transform(df['title'])
count_vec_text = count_model_text.fit_transform(df['text'])

X = [[count_vec_title, count_vec_text]]
y = df['label']

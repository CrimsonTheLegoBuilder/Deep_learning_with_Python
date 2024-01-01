import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import re

import urllib.request
urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', 'shopping.txt')

raw = pd.read_table('shopping.txt', names=['rating', 'review'])
# print(raw)

raw['label'] = np.where(raw['rating'] > 3, 1, 0)
# print(raw)

raw['review'] = raw['review'].str.replace("[^ㄱ-ㅣ가-힣0-9 ]", "", regex=True)
# raw['review'] = raw['review'].apply(lambda x: re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s]', '', x))
# print(raw)
# print(raw['review'])
# print(raw.isnull().sum())

raw.drop_duplicates(subset=['review'], inplace=True)
print(raw)

unique_txt = raw['review'].tolist()
unique_txt = ''.join(unique_txt)
unique_txt = list(set(unique_txt))
unique_txt.sort()
# print(unique_txt[0:100])

tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
raw_list = raw['review'].tolist()
tokenizer.fit_on_texts(raw_list)
print(tokenizer.word_index)
print(raw_list[0:10])

train_seq = tokenizer.texts_to_sequences(raw_list)
print(train_seq[0:10])

Y = raw['label'].tolist()
print(Y[0:10])

raw['length'] = raw['review'].str.len()

print(raw.head())
print(raw.describe())

print(raw['length'][raw['length'] < 100].count())

from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(train_seq, maxlen=100)

from sklearn.model_selection import train_test_split
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42)

print(len(trainX))
print(len(valX))

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len() + 1, 16),

])
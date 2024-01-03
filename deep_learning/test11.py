import pandas as pd
import tensorflow as tf

import test11

data = pd.read_csv('train.csv')

# print(data)
# print(data.isnull().sum())
avg = data['Age'].mean()
# print(avg)
mod = data['Embarked'].mode()
data['Age'].fillna(value=30, inplace=True)
data['Embarked'].fillna(value='S', inplace=True)
# print(data)

ans = data.pop('Survived')

ds = tf.data.Dataset.from_tensor_slices((dict(data), ans))

print(ds)

feature_columns = []


def normalizer(x):
    minn = data['Fare'].min()
    maxx = data['Fare'].max()
    return (x - minn) / (maxx - minn)


# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked
# origin  Fare Parch SibSp : numeric_column
# simple  Age : bucketized_column
# categories  Sex Embarked Pclass : indicator_column
# many categories  Ticket : embedding_column
feature_columns.append(tf.feature_column.numeric_column('Fare', normalizer_fn=normalizer()))
feature_columns.append(tf.feature_column.numeric_column('Parch'))
feature_columns.append(tf.feature_column.numeric_column('SibSp'))

Age = tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10, 20, 30, 40, 50, 60])
feature_columns.append(Age_bucket)

# print(Age)
# print(feature_columns)
vocab = data['Sex'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

vocab = data['Embarked'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

vocab = data['Pclass'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Pclass', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

#embedding
vocab = data['Ticket'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', vocab)
one_hot = tf.feature_column.embedding_column(cat, dimension=9)
feature_columns.append(one_hot)

ds_batch = ds.batch(32)
next(iter(ds_batch))[0]
feature_layer = tf.keras.layers.DenseFeatures(tf.feature_column.numeric_column('Fare'))
feature_layer(next(iter(ds_batch))[0])

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

ds_batch = ds.batch(32)
model.fit(ds_batch, shuffle=True, epochs=20)

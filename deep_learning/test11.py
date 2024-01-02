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
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked
# origin  Fare Parch SibSp : numeric_column
# simple  Age : bucketized_column
# categories  Sex Embarked Pclass : indicator_column
# many categories  Ticket : embedding_column

feature_columns.append(tf.feature_column.numeric_column('Fare'))
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




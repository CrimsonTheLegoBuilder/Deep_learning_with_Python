import os
# os.environ['KAGGLE_CONFIG_DIR'] = '/content/'
# !kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
# !unzip -q train.zip -d .
import os
import tensorflow as tf
import shutil
from tensorflow.keras.applications.inception_v3 import InceptionV3

# for i in os.listdir('./content/train/'):
#     if 'cat' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/cat/' + i)
#     if 'dog' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/dog/' + i)
#

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './content/dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './content/dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=1234
)


def pretreatment(i_, ans):
    i_ = tf.cast(i_/255.0, tf.float32)
    return i_, ans


train_ds = train_ds.map(pretreatment)
val_ds = val_ds.map(pretreatment)

inception_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
inception_model.load_weights('inception_v3.h5')

# inception_model.summary()

for i in inception_model.layers:
    i.trainable = False

unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6':
        unfreeze = True
    if unfreeze:
        i.trainable = True
    # i.trainable = False

last = inception_model.get_layer('mixed9')
print(last)
print(last.output)
print(last.output_shape)

layer1 = tf.keras.layers.Flatten()(last.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)

model = tf.keras.Model(inception_model.input, layer3)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(train_ds, validation_data=val_ds, epochs=2)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, epochs=2)

# os.mkdir('./content/dataset')
# os.mkdir('./content/dataset/cat')
# os.mkdir('./content/dataset/dog')
#
# for i in os.listdir('./content/train/'):
#     if 'cat' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/cat/' + i)
#     if 'dog' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/dog/' + i)
#
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     '/content/dataset/',
#     image_size=(150, 150),
#     batch_size=64,
#     subset='training',
#     validation_split=0.2,
#     seed=1234
# )
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     '/content/dataset/',
#     image_size=(150, 150),
#     batch_size=64,
#     subset='validation',
#     validation_split=0.2,
#     seed=1234
# )
#
# print(train_ds)
#
#
# def pretreatment(i_, ans):
#     i_ = tf.cast(i_/255.0, tf.float32)
#     return i, ans
#
#
# train_ds = train_ds.map(pretreatment)
# val_ds = val_ds.map(pretreatment)

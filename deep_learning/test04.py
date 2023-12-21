import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator1 = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    # vertical_flip=False,
    fill_mode='nearest',
    # featurewise_center=False,
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # zca_whitening=False,
    # zca_epsilon=1e-06,
    # brightness_range=None,
    # channel_shift_range=0.0,
    # cval=0.0,
    # preprocessing_function=None,
    # data_format=None,
    # validation_split=0.0,
    # interpolation_order=1,
    # dtype=None
)
trainer = generator1.flow_from_directory(
    './content/dataset',
    target_size=(64, 64),
    seed=123,
    color_mode='rgb',
    batch_size=64,
    class_mode='binary',
    shuffle=True
)

generator2 = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)
validator = generator1.flow_from_directory(
    './content/dataset',
    target_size=(64, 64),
    seed=123,
    color_mode='rgb',
    batch_size=64,
    class_mode='binary',
    shuffle=True
)

# os.mkdir('./content/dataset')
# os.mkdir('./content/dataset/cat')
# os.mkdir('./content/dataset/dog')

# for i in os.listdir('./content/train/'):
#     if 'cat' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/cat/' + i)
#     if 'dog' in i:
#         shutil.copyfile('./content/train/' + i, './content/dataset/dog/' + i)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './content/dataset/',
    image_size=(64, 64),
    batch_size=32,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './content/dataset/',
    image_size=(64, 64),
    batch_size=32,
    subset='validation',
    validation_split=0.2,
    seed=1234
)


def f(i_, ans_):
    i_ = tf.cast(i_/255.0, tf.float32)
    return i_, ans_


train_ds = train_ds.map(f)
val_ds = val_ds.map(f)
# print(train_ds)
#
for i, ans in train_ds.take(1):
    print(i)
    print(ans)
#     plt.imshow(i[0].numpy().astype('uint8'))
#     plt.show()

model = tf.keras.Sequential([

    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()  # 요약 보기

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(
    # train_ds,
    trainer,
    validation_data=val_ds,
    epochs=5
)

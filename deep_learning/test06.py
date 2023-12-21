import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# print(trainX.shape)
# print(trainY)

# plt.imshow(trainX[2])
# plt.gray()
# plt.colorbar()
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirts', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()  # 요약 보기

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./logs/{}'.format('conv1' + str(int(time.time()))))

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[tensorboard])


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()  # 요약 보기

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

import time
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

tensorboard = TensorBoard(log_dir='./logs/{}'.format('conv2' + str(int(time.time()))))

# es = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[tensorboard, es])

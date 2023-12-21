import tensorflow as tf
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

f = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint/mnist',
    monitor='val_acc',
    mode='max',
    sve_weights_only=True,
    save_freq='epoch'
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_crossentropy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[f])

# model.save('./new_folder/model1')
# l_model = tf.keras.models.load_model('./new_folder/model1')
# l_model.summary()
# l_model.evaluate(testX, testY)

# model2 = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax'),
# ])
#
# # f = tf.keras.callbacks.ModelCheckpoint(
# #     filepath='./checkpoint/mnist',
# #     monitor='val_acc',
# #     mode='max',
# #     sve_weights_only=True,
# #     save_freq='epoch'
# # )
#
# model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # model.save('./new_folder/model1')
# # l_model = tf.keras.models.load_model('./new_folder/model1')
# # l_model.summary()
# # l_model.evaluate(testX, testY)
# model2.load_weights('./checkpoint/mnist')
#
# model2.evaluate(testX, testY)


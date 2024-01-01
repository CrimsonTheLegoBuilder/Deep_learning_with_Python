import tensorflow as tf
import numpy as np

premodel = tf.keras.models.load_model('model1')

text = open('pianoabc.txt', 'r').read()
# print(text)

unique_txt = list(set(text))
unique_txt.sort()

t2n = {}
n2t = {}

for i, data in enumerate(unique_txt):
    t2n[data] = i
    n2t[i] = data

no_txt = []
for i in text:
    no_txt.append(t2n[i])

first = no_txt[117:117+25]
first = tf.one_hot(first, 31)
first = tf.expand_dims(first, axis=0)
# print(first)

music = []

for i in range(200):
    prediction = premodel.predict(first)
    prediction = np.argmax(prediction[0])

    new_pre = np.random.choice(unique_txt, 1, p=prediction[0])
    # print(prediction)
    # print(no_txt[117+25])

    music.append(prediction)

    nxt_input = first.numpy()[0][1:]
    # print(nxt_input)

    one_hot_num = tf.one_hot(prediction, 31)
    # print('원핫한거', one_hot_num)

    first = np.vstack([nxt_input, one_hot_num.numpy()])
    first = tf.expand_dims(first, axis=0)

# print(music)

m2t = []

for i in music:
    m2t.append(n2t[i])

print(''.join(m2t))
# X = []
# Y = []
#
# for i in range(0, len(no_txt) - 25):
#     X.append(no_txt[i:i+25])
#     Y.append(no_txt[i + 25])
#
# X = tf.one_hot(X, 31)
# Y = tf.one_hot(Y, 31)
#
#
# model = tf.keras.models.Sequential([
#     # tf.keras.layers.LSTM(100, input_shape=(25, 31), return_sequences=True),
#     tf.keras.layers.LSTM(100, input_shape=(25, 31)),
#     tf.keras.layers.Dense(31, activation='softmax')
# ])
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X, Y, batch_size=64, epochs=30, verbose=2)
#
# model.save('model1')
import numpy as np
import tensorflow as tf

text = open('pianoabc.txt', 'r').read()
# print(text)

unique_txt = list(set(text))
unique_txt.sort()

t2n = {}
n2t = {}

for i, data in enumerate(unique_txt):
    t2n[data] = i
    n2t[i] = data

# print(t2n['A'])
no_txt = []
for i in text:
    no_txt.append(t2n[i])

# print(no_txt)

X = []
Y = []

for i in range(0, len(no_txt) - 25):
    X.append(no_txt[i:i+25])
    Y.append(no_txt[i + 25])

print(X[0:5])
print(Y[0:5])

print(np.array(X).shape)
X = tf.one_hot(X, 31)
Y = tf.one_hot(Y, 31)
print(X[0:2])

model = tf.keras.models.Sequential([
    # tf.keras.layers.LSTM(100, input_shape=(25, 31), return_sequences=True),
    tf.keras.layers.LSTM(100, input_shape=(25, 31)),
    tf.keras.layers.Dense(31, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=64, epochs=30, verbose=2)

model.save('model1')

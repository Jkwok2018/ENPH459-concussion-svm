import numpy as np
from numpy.lib.function_base import average
import scipy.io

# NN packages
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


eeg_file = scipy.io.loadmat('/Users/melodyzhao/Desktop/Python/ENPH459-concussion-svm/New_Code/FeatureMatrix.mat')
fMatrix = eeg_file["data"].tolist()
labels = list(map(int, eeg_file["labels"][0].tolist()))

tf.random.set_seed(5)
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(756,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.add(Dense(128, input_dim=756, activation='relu'))

# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(fMatrix, labels, epochs=300, batch_size=20)
_, accuracy = model.evaluate(fMatrix, labels)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(fMatrix)
print(predictions)



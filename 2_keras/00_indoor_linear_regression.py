# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:39:48 2019

@author: zouco
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# a typical Keras approach is bulid -> compile -> fit -> predict
#  model=Sequential; model.add() ....
#  model.compile(optimizer=, loss=, metrics=)
#  model.fit(...)
#  model.predict(...)


# for example, a linear model:
# -------------------------------------------------------------------
# 01 linear model:
# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points


# start
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.complie(optimizer="sgd", loss="mse")

for i in range(301):
    history = model.fit(X_train, Y_train, batch_size=100, epochs=3)
    # or model.train_on_batch(x,y), it is more recommended
    print(history.history['loss'])

model.evaluate(X_test, Y_test, batch_size=10)
W, b = model.layers[0].get_weights()


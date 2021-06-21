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

from material.data import *

# a typical Keras approach is bulid -> compile -> fit -> predict
#  model=Sequential; model.add() ....
#  model.compile(optimizer=, loss=, metrics=)
#  model.fit(...)
#  model.predict(...)

x_train, y_train, x_test, y_test = trival_data2()

# plot data
plt.scatter(x_train, y_train)
plt.show()

# start
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(optimizer="sgd", loss="mse")

# train
model.fit(x_train, y_train, batch_size=100, epochs=3)

# predict
Y_pred = model.predict(x_test)




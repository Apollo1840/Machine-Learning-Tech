# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:39:48 2019

@author: zouco
"""
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

from material.data import *

# a typical Keras approach is bulid -> compile -> fit -> predict
#  model=Sequential(); model.add() ....
#  model.compile(optimizer=, loss=, metrics=)
#  model.fit(...)
#  model.predict(...)

x_train, y_train, x_test, y_test = mnist_data()

# step 1: build model
# Sequential way to build model:

# type 1:
model = Sequential([
    Dense(16, input_shape=(784,), activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="softmax")
])

del model

# type 2:
model = Sequential()
model.add(Dense(16, input_shape=(784,), activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(2, activation="softmax"))

del model

# Funtional way to build model
x = Input(shape=(784,))
h = Dense(16)(x)
h = Dense(2)(h)
y = Dense(2)(h)
model = Model(inputs=x, outputs=y)

# step 2: complie model
# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

# step 3: fit model
# train the model
history = model.fit(x_train, y_train, epochs=2, batch_size=32)

# consider validation
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=5,
                    validation_split=0.2)

# consider validation
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=5,
                    validation_data=(x_train[-100:], y_train[-100:]))

# print loss
print(history.history['loss'][-1])

# step 4: model predict
y_pred = model.predict(x_test[-100:])

# evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

# knowledge 02: get weights
W, b = model.layers[0].get_weights()

# save and load
import os
import json
from keras.models import load_model
from keras.models import model_from_json
PROJECT_FOLDER = os.path.join(os.getcwd(), "material/storage")
os.chdir(PROJECT_FOLDER)
print(os.getcwd())


# way 1:
model.save("dummie_model.h5")
model = load_model("dummie_model.h5")

# way 2, save model shape and weights differently:

# save
json_string = model.to_json()
with open("dummie_model.json", "w") as f:
    json.dump(json_string, f)

model.save_weights("dummie_model_weights.h5")

# load
with open("dummie_model.json", "r") as f:
    model = model_from_json(json.load(f))

model.load_weights("dummie_model_weights.h5")
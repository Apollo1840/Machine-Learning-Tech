# Import all the required Libraries
import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, \
    Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU


class CAE():

    def __init__(self, **kwargs):
        print("init")
        self.model, self.encoder, self.decoder = ConvolutionalAutoEncoder()
        self.a = 10

    def fit(self, x_train, y_train, **kwargs):
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(x_train, y_train, **kwargs)

    def predict(self, x_test, **kwargs):
        return self.encoder.predict(x_test, **kwargs)

    def reconstruct(self, x_test, **kwargs):
        return self.model.predict(x_test, **kwargs)


def conv_encoder():
    # ENCODER
    inp = Input((28, 28, 1))
    e = Conv2D(32, (3, 3), activation='relu')(inp)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(64, (3, 3), activation='relu')(e)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(64, (3, 3), activation='relu')(e)
    l = Flatten()(e)
    l = Dense(49, activation='softmax')(l)

    return Model(inp, l)


def conv_decoder():
    # DECODER
    l = Input((49,))
    d = Reshape((7, 7, 1))(l)
    d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(d)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

    return Model(l, decoded)


def ConvolutionalAutoEncoder():
    encoder = conv_encoder()
    decoder = conv_decoder()

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model, encoder, decoder

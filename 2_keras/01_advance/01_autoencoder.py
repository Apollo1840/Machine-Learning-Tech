# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:39:48 2019

@author: zouco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from keras.datasets import mnist

from material.data import *
from ..modelbank.models.autoencoder_cnn import CAE


def t_sne_visualize(x, y):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj = tsne.fit_transform(x)

    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'classes': y})
    sns.scatterplot(x="X", y="Y",
                    hue="classes",
                    legend='full',
                    size=0.5,
                    alpha=0.2,
                    data=tsne_df)
    plt.show()


# create a plot of generated images (reversed grayscale)
def show_plot(examples, n, with_channel=True):
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        if with_channel:
            # shape = (n_sample, x_axis, y_axis, channel)
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        else:
            # shape = (n_sample, x_axis, y_axis)
            plt.imshow(examples[i], cmap='gray_r')

    plt.show()


# load data
x_train, y_train, x_test, y_test = mnist_data()
t_sne_visualize(x_test.reshape(x_test[:500].shape[0], -1), [np.argmax(y) for y in y_test[:500]])

# train model
model = CAE()
model.model.summary()
model.fit(x_train, x_train, epochs=10, validation_split=0.2)

# eval model
x_train_r = model.reconstruct(x_train[:10])

show_plot(x_train, 3)
show_plot(model.reconstruct(x_train[:10]), 3)

# analysis model
xcode = model.predict(x_test[:500])
t_sne_visualize(xcode.reshape(xcode.shape[0], -1), [np.argmax(y) for y in y_test[:500]])

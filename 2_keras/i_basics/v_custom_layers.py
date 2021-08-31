# -*- coding: utf-8 -*-
"""
# 自定义层

doc:
https://keras.io/zh/layers/writing-your-own-keras-layers/

good examples
https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/

very good examples:
https://zhuanlan.zhihu.com/p/36436904

https://blog.csdn.net/u013084616/article/details/79295857

https://www.youtube.com/results?search_query=keras+custom+layer
"""
import numpy as np

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Layer, Lambda, Conv2D, BatchNormalization
from keras import backend as K

model = Sequential()

# example 1
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))


# example 2

# add a layer that returns
# the concatenation of
# the positive part of the input
# and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)


def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)


model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))


# layer inherit from tf.keras.layer.Layer
class MyDenseLayer(Layer):
    # basicall the same as Dense layer without activation
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


layer = MyDenseLayer(10)


# multiple inputs:

class mul(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        inp1, inp2 = inputs
        Z = inp1 * inp2
        return Z


# block inherite from tf.keras.Model
# benefits:
# - block.fit(), block.evaluate(), block.save()
# - block tracks internal layers

class ResnetIdentityBlock(Model):
    def __init__(self, kernel_size, n_filters):
        # convention: specify layer parameters in __init__()

        super(ResnetIdentityBlock, self).__init__(name='')
        n_filters1, n_filters2, n_filters3 = n_filters

        self.conv2a = Conv2D(n_filters1, (1, 1))
        self.bn2a = BatchNormalization()

        self.conv2b = Conv2D(n_filters2, kernel_size, padding='same')
        self.bn2b = BatchNormalization()

        self.conv2c = Conv2D(n_filters3, (1, 1))
        self.bn2c = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

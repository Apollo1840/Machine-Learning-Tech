import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Conv1D,
    Activation,
    MaxPooling1D,
    Input)
from keras.models import Model, Sequential
import keras.backend as K


class DActivation(Activation):
    def observation_field(self, field):
        return field


class ReverseBiasLayer(Layer):
    # basicall the same as Dense layer without activation
    def __init__(self, bias):
        super(ReverseBiasLayer, self).__init__()
        self.bias = K.constant((-1) * bias)

    def call(self, inputs, **kwargs):
        return inputs + self.bias

    def observation_field(self, field):
        return field


class Deconv():

    @staticmethod
    def get_observation1d(field, kernel_size, stride, padding_size, observe_size):
        """
        get observation field base on a given field, and some convLayer parameters

        :param field: Tuple, (start, end): eg:  (1,5), it means a stripe from fmap[1] to fmap[5]
        :param: kernel_size: int
        :param: stride: int
        :param: padding_size: int
        :param: observe_size: int, the input length of the convLayer. as maximum of observation loc
        """

        start, end = field
        affine_sta = lambda x: max(0, stride * x - padding_size)
        affine_end = lambda x: min(observe_size, stride * x + kernel_size - 1 - padding_size)
        return affine_sta(start), affine_end(end)

    @staticmethod
    def get_observation2d(field, kernel_size, stride, padding_size, observe_size):
        """
        get observation field base on a given field, and some convLayer parameters

        :param field: Tuple[Tuple], (x,y of top_left block, x,y of bottom_right block): eg:  ((1,2),(4,5))
            it means a box from fmap[1, 2] to fmap[4, 5]
        :param: kernel_size: int
        :param: stride: int
        :param: padding_size: int
        :param: observe_size: int, the input square size of the convLayer. as maximum of observation loc
        """

        top_left, bottom_right = field

        affine_tl = lambda x: max(0, stride * x - padding_size)
        top_left_new = (affine_tl(top_left[0]), affine_tl(top_left[1]))

        affine_br = lambda x: min(observe_size, stride * x + kernel_size - 1 - padding_size)
        bottom_right_new = (affine_br(bottom_right[0]), affine_br(bottom_right[1]))

        return top_left_new, bottom_right_new


class DeConv1D(Conv1D, Deconv):

    def __init__(self, *args, **kwargs):
        super(DeConv1D, self).__init__(*args, **kwargs)

        if self.strides == 1 and self.padding == "same":
            self.pad_size = self.kernel_size[0] // 2  # needed for calculate observation_field
        else:
            self.pad_size = 0

    def observation_field(self, field):
        return super().get_observation1d(field,
                                         kernel_size=self.kernel_size,
                                         stride=self.strides,
                                         padding_size=1,
                                         observe_size=self.output_shape[1])  # None, fmap_size, n_channels


class MaxUnPooling1D(Layer, Deconv):
    # basicall the same as Dense layer without activation
    # assert pool_size == strides
    def __init__(self, switch_matrix, pool_size):
        super(MaxUnPooling1D, self).__init__()
        self.switch_matrix = K.constant(switch_matrix)
        self.pool_size = pool_size

    def call(self, inputs, **kwargs):
        return tf.repeat(inputs, 2, axis=1) * self.switch_matrix

    def observation_field(self, field):
        return super().get_observation1d(field,
                                         kernel_size=self.pool_size,
                                         stride=self.pool_size,
                                         padding_size=0,
                                         observe_size=float("Inf"))


class DeConv1DModel(Sequential):

    @classmethod
    def from_conv1d(cls, model, n_layers, current_feature_maps, lyid_feature_maps):
        deconv_layers = cls()
        deconv_layers.add(Input(shape=model.layers[n_layers].output_shape[1:]))
        for i in range(n_layers, 0, -1):
            if isinstance(model.layers[i], Conv1D):
                deconv_layers.add(cls.get_activation())
                deconv_layers.add(cls.get_deconv1d_reverse_bias(model.layers[i]))
                deconv_layers.add(cls.get_deconv1d(model.layers[i]))
            if isinstance(model.layers[i], MaxPooling1D):
                deconv_layers.add(
                    cls.get_maxunpool1d(model.layers[i], current_feature_maps, lyid_feature_maps.index(i)))
        return deconv_layers

    def observation_field(self, loc):
        observation_field = (loc, loc)
        for i_layer, layer in enumerate(self.layers):
            observation_field = layer.observation_field(observation_field)
        return observation_field

    @staticmethod
    def get_deconv1d_reverse_bias(conv1d_layer: Conv1D):
        return ReverseBiasLayer(conv1d_layer.get_weights()[1])

    @staticmethod
    def get_deconv1d(conv1d_layer: Conv1D):
        W = conv1d_layer.get_weights()[0]
        # W: kernel_width, kernel_depth, n_filters

        # Reverse the conv operation
        W = np.transpose(W, (0, 2, 1))
        # Transpose the columns and rows
        W = W[::-1, :, :]

        n_filters = W.shape[2]
        kernel_size = W.shape[0]
        strides = conv1d_layer.strides
        padding = conv1d_layer.padding
        b = np.zeros(n_filters)

        return DeConv1D(n_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_initializer=tf.constant_initializer(W),
                        bias_initializer=tf.constant_initializer(b),
                        trainable=False)

    @staticmethod
    def get_maxunpool1d(maxpool_layer: MaxPooling1D, current_feature_maps, id_layer):
        strides = maxpool_layer.strides
        pool_size = maxpool_layer.pool_size
        assert strides == pool_size  # otherwise it will be very hard to implement
        size = pool_size[0]

        current_fmap = current_feature_maps[id_layer][0]
        previous_fmap = current_feature_maps[id_layer - 1][0]
        assert current_fmap.shape[0] == previous_fmap.shape[0] // size

        switch_matrix = DeConv1DModel.switch_matrix_1d(current_fmap, previous_fmap, size)
        return MaxUnPooling1D(switch_matrix, maxpool_layer.pool_size)

    @staticmethod
    def max_mask(x):
        mask = np.zeros(np.prod(x.shape))
        mask[np.argmax(x)] = 1
        mask = np.reshape(mask, x.shape)
        return mask

    @staticmethod
    def switch_matrix_1d(fmap, vmap, size):
        switch_matrix = []
        for i in range(fmap.shape[0]):
            switch_locs = []
            for j in range(fmap.shape[1]):
                max_pool_field = vmap[i * size: (i + 1) * size, j]
                switch_locs.append(DeConv1DModel.max_mask(max_pool_field))
            switch_matrix.append(switch_locs)

        # switch_matrix : length, channel, pool
        switch_matrix = np.array(switch_matrix)
        n_channels = switch_matrix.shape[1]
        mask_by_channels = [np.concatenate(switch_matrix[:, i, :], axis=-1) for i in range(n_channels)]
        switch_matrix = np.transpose(mask_by_channels, (1, 0))
        return switch_matrix

    @staticmethod
    def test(model, feature_maps):
        # test deconv layers
        d2d = DeConv1DModel.get_deconv1d(model.layers[1])
        rb = DeConv1DModel.get_deconv1d_reverse_bias(model.layers[1])

        test_model = Sequential()
        test_model.add(Input(shape=model.layers[1].output_shape[1:]))
        test_model.add(rb)
        test_model.add(d2d)
        test_model.summary()

        sig_re = test_model.predict(feature_maps[0])[0]
        plt.plot(sig_re)


def get_fmap_model(model, ixs_layers=(1, 3, 4, 6, 7, 9)):
    outputs = [model.layers[i].output for i in ixs_layers]
    fm_model = Model(inputs=model.inputs, outputs=outputs)
    model.trainable = False
    return fm_model


def get_deconv_model(model, n_layers, current_feature_maps, lyid_feature_maps):
    lyid_feature_maps = list(lyid_feature_maps)
    return DeConv1DModel.from_conv1d(model, n_layers, current_feature_maps, lyid_feature_maps)


def plot_feature_maps(feature_maps, square=4):
    for fmap in feature_maps:
        print(fmap.shape)
        # plot all 64 maps in an 8x8 squares
        ix = 0
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix + 1)
                ax.set_xticks([])
                plt.plot(fmap[0, :, ix])
                ix += 1
        # show the figure
        plt.show()


def filter_fmap(fmap, i_filter, return_loc=True):
    fmap_filter = fmap[0][:, i_filter]
    print(fmap_filter.shape)

    max_ = np.argmax(fmap_filter)
    fmap_filter_mask = np.zeros(fmap_filter.shape[0])
    fmap_filter_mask[max_] = 1

    fmap_mask = np.zeros(fmap.shape)
    fmap_mask[0, :, i_filter] = fmap_filter_mask

    fmap_filtered = fmap * fmap_mask
    print(fmap_filtered.shape)

    if return_loc:
        return fmap_filtered, max_
    else:
        return fmap_filtered


def show_reconstruct_pattern(deconv_model: DeConv1DModel, fmap, img, i_filter=None):
    # highest activati within all fmaps
    if i_filter is None:
        max_ = np.argmax(fmap[0])
        i_filter = max_ % fmap[0].shape[-1]

    fmap_filtered, max_loc = filter_fmap(fmap, i_filter)

    tl, br = deconv_model.observation_field(max_loc)
    re_act = deconv_model.predict(fmap_filtered)[0]
    re_act_focus = re_act[tl[0]:br[0], tl[1]:br[1], :]

    plt.plot(re_act)
    plt.plot(re_act_focus)
    plt.plot(img[tl[0]:br[0], tl[1]:br[1], :])

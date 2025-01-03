
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization, add, Input, AveragePooling2D
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop


def residualBlock(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x1 = Activation("relu")(conv(inputs))
    x2 = Activation("relu")(add([conv(x1), inputs]))
    return x2


def residualBlock_trival(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  activation="relu")

    x1 = conv(inputs)
    x = add([inputs, x1])
    return x


def residualBlock_trival2(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  activation="relu")

    x1 = conv(inputs)
    x2 = conv(x1)
    x = add([inputs, x2])
    return x


def resNet(input_shape, output_shape):

    x = Input(shape=input_shape)
    h = residualBlock(x)
    for _ in range(9):
        h = residualBlock(x)

    h = Flatten()(h)
    h = Dense(126)(h)
    h = Dense(30)(h)
    y = Dense(output_shape, activation="softmax", kernel_initializer='he_normal')(h)

    model = Model(inputs=x, outputs=y)

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def conv_resnet(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = conv_resnet(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for id_res_block in range(num_res_blocks):
            x, num_filters_out = res_block(id_res_block, x, stage, num_filters_in)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def res_block(res_block, x, stage, num_filters_in):
    activation_first_conv = 'relu'
    normalize_first_conv = True
    strides_first_conv = 1
    if stage == 0:
        num_filters_out = num_filters_in * 4
        if res_block == 0:  # first layer and first stage
            activation_first_conv = None
            normalize_first_conv = False
    else:
        num_filters_out = num_filters_in * 2
        if res_block == 0:  # first layer but not first stage
            strides_first_conv = 2  # downsample

    # bottleneck residual unit
    y = conv_resnet(inputs=x,
                    num_filters=num_filters_in,
                    kernel_size=1,
                    strides=strides_first_conv,
                    activation=activation_first_conv,
                    batch_normalization=normalize_first_conv,
                    conv_first=False)
    y = conv_resnet(inputs=y,
                    num_filters=num_filters_in,
                    kernel_size=3,
                    strides=1,
                    batch_normalization=True,
                    conv_first=False)
    y = conv_resnet(inputs=y,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    conv_first=False)
    if res_block == 0:
        # linear projection residual shortcut connection to match
        # changed dims
        x = conv_resnet(inputs=x,
                        num_filters=num_filters_out,
                        kernel_size=1,
                        strides=strides_first_conv,
                        activation=None,
                        batch_normalization=False)
    x = keras.layers.add([x, y])
    return x, num_filters_out


def res_block_easy(stage, res_block, x, num_filters_in):
    # equal to res_block

    if stage == 0:
        if res_block == 0:
            activation_first_conv = None
            normalize_first_conv = False
            strides_first_conv = 1
            num_filters_out = num_filters_in * 4
        else:
            activation_first_conv = 'relu'
            normalize_first_conv = True
            strides_first_conv = 1
            num_filters_out = num_filters_in * 4
    else:
        if res_block == 0:
            activation_first_conv = 'relu'
            normalize_first_conv = True
            strides_first_conv = 2
            num_filters_out = num_filters_in * 2
        else:
            activation_first_conv = 'relu'
            normalize_first_conv = True
            strides_first_conv = 1
            num_filters_out = num_filters_in * 2

    # bottleneck residual unit
    y = convs_resnet(x, num_filters_in, num_filters_out, strides_first_conv, activation_first_conv,
                     normalize_first_conv)

    if res_block == 0:
        # linear projection residual shortcut connection to match
        # changed dims
        x = conv_resnet(inputs=x,
                        num_filters=num_filters_out,
                        kernel_size=1,
                        strides=strides_first_conv,
                        activation=None,
                        batch_normalization=False)

    x = keras.layers.add([x, y])
    return x, num_filters_out


def convs_resnet(x, num_filters_in, num_filters_out, strides_first_conv, activation_first_conv, normalize_first_conv):
    y = conv_resnet(inputs=x,
                    num_filters=num_filters_in,
                    kernel_size=1,
                    strides=strides_first_conv,
                    activation=activation_first_conv,
                    batch_normalization=normalize_first_conv,
                    conv_first=False)

    y = conv_resnet(inputs=y,
                    num_filters=num_filters_in,
                    kernel_size=3,
                    strides=1,
                    activation="relu",
                    batch_normalization=True,
                    conv_first=False)

    y = conv_resnet(inputs=y,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=1,
                    activation="relu",
                    batch_normalization=True,
                    conv_first=False)

    return y

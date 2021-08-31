import tensorflow as tf
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Masking, Lambda, LSTM

raw_inputs = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

padded_inputs = pad_sequences(raw_inputs, padding="post")
print(padded_inputs)

# ------------------------------------------------------------------
# method 1: use support masking layer
embedding = Embedding(input_dim=5000, output_dim=16)
masked_output = embedding(padded_inputs)

print(masked_output._keras_mask)

# ------------------------------------------------------------------
# method 2: use masking layer

# Simulate the embedding lookup by expanding the 2D input to 3D,
# with embedding dimension of 10.
SimEmbedding = Lambda(lambda inputs: tf.cast(
    tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, 10]), tf.float32
))

unmasked_embedding = SimEmbedding(padded_inputs)
print(unmasked_embedding)

masking_layer = Masking()
masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)


# -------------------------------------------------------------------
# method 3: use masking layer and pass the mask to custom layer who should accept mask in __call__()

class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = SimEmbedding
        self.masking = Masking()
        self.lstm = LSTM(32, return_sequences=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        # Note that you could also prepare a `mask` tensor manually.
        # It only needs to be a boolean tensor
        # with the right shape, i.e. (batch_size, timesteps).
        mask = self.masking.compute_mask(x)
        output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
        return output


layer = MyLayer()
layer(padded_inputs)


# custom masking layer

class CustomMasking(Masking):
    # this one can only be used in method 3.
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.masking = Masking()
        super(CustomMasking, self).__init__(**kwargs)

    def compute_mask(self, inputs):
        """

        :param: inputs: [batch, token]
        """
        expand_mask = tf.cast(
            tf.tile(tf.expand_dims(inputs, axis=-1), [1, 1, self.output_dim]), tf.float32
        )

        return self.masking.compute_mask(expand_mask)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


layer = CustomMasking(16)
layer.compute_mask([1, 1, 1, 0, 0, 0])


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = Embedding(input_dim=5000, output_dim=16)
        self.masking = CustomMasking(16)
        self.lstm = LSTM(32, return_sequences=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        mask = self.masking.compute_mask(inputs)
        output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
        return output

layer = MyLayer()
layer(padded_inputs)
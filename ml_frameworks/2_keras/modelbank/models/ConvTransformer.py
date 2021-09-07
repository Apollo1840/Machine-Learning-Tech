from keras.models import Sequential
from keras.layers import (Layer,
                          Input,
                          Dense,
                          LayerNormalization,
                          Dropout,
                          Embedding,
                          Conv1D,
                          MaxPooling1D,
                          Flatten,
                          TimeDistributed,
                          GlobalAveragePooling1D,
                          MultiHeadAttention)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        """

                |
        +--------------------+
        |   multi-head-att  |
        +-------------------+
                |
        +------------------------+
        |   dropout & layernorm  |
        +------------------------+
                |
        +--------------------+
        |   fully connected  |
        +-------------------+
                |
        +------------------------+
        |   dropout & layernorm  |
        +------------------------+
                |

        """

        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.layernorm2(out1 + ffn_output)
        return ffn_output


def ConvTransformer():
    # define CNN Layers
    cnn = Sequential(name="ConvLayers")
    cnn.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(28, 1)))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Flatten())
    cnn.add(Dense(32, activation="relu"))

    # define LSTM model
    model = Sequential(name="ConvTransformer")

    model.add(Input(shape=(28, 28, 1)))
    model.add(TimeDistributed(cnn))

    model.add(TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

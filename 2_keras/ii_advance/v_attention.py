import keras
from keras.layers import (Input,
                          GRU,
                          Activation,
                          Concatenate,
                          Dense,
                          TimeDistributed,
                          Lambda,
                          Flatten,
                          Embedding)
from keras.models import Model
import keras.backend as K
from keras_self_attention import SeqSelfAttention


def GRUAtt(vocab_size, max_len):
    """
    Attention without linear combination.

    Attention only works as a weighting mechanism for tokens.


    """
    input_ = Input(shape=(max_len,))
    words = Embedding(vocab_size, 100, input_length=max_len)(input_)
    sen = GRU(64, return_sequences=True)(words)  # [b_size,maxlen,64]

    # attention
    att = Dense(1, activation="softmax", name='attention_vec')(sen)  # [b_size,maxlen,1]
    z = Lambda(lambda x: x[0] * x[1])([att, sen])

    output = Flatten()(z)
    output = Dense(32, activation="relu")(output)
    output = Dense(2, activation='softmax')(output)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    return model


def GRUAttMulti(vocab_size, max_len, hidden_dim=64):
    """
    Attention without linear combination.

    Attention only works as a weighting mechanism for tokens.
    but every dimension has different weight coefficients.


    """
    input_ = Input(shape=(max_len,))
    words = Embedding(vocab_size, 100, input_length=max_len)(input_)
    sen = GRU(hidden_dim, return_sequences=True)(words)  # [b_size,maxlen,64]

    # attention
    att = Dense(hidden_dim, activation="softmax", name='attention_vec')(sen)  # [b_size,maxlen,1]
    z = Lambda(lambda x: x[0] * x[1])([att, sen])

    output = Flatten()(z)
    output = Dense(32, activation="relu")(output)
    output = Dense(2, activation='softmax')(output)
    model = Model(inputs=input_, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

    return model


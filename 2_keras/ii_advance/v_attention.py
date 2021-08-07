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


def GRUAtt(vocab_size, max_len):
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
    model.summary()


def some_attention():
    ncoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)

    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

    attn_layer = Activation(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

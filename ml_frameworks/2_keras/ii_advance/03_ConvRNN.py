from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense


def ConvRNN():
    # define CNN model
    cnn = Sequential()
    cnn.add(Conv2D(1, (2, 2), activation='relu', padding='same', input_shape=(10, 10, 1)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())

    # define LSTM model
    model = Sequential()
    model.add(TimeDistributed(cnn))
    model.add(LSTM(4, return_sequences=True, return_state=True))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model

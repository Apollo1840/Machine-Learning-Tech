import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sacred import Experiment
from sacred.observers import MongoObserver
from pymongo import MongoClient

# set up your mongodb
# > sudo apt-get install mongodb
# > sudo service mongodb start
# > sduo service mongodb status

# > sudo npm install -g omniboard
# > omniboard -m localhost:27017:sacred
# more info: https://vivekratnavel.github.io/omniboard/#/quick-start

ex = Experiment()
ex.observers.append(MongoObserver())

def load_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


@ex.config
def config():
    input_shape = (28, 28, 1)
    num_classes = 10
    hidden_shape = (32, 64)
    batch_size = 128
    epochs = 3


@ex.capture
def basic_cnn(input_shape=(28, 28, 1), num_classes=10, hidden_shape=(32, 64)):
    model = keras.Sequential(
        [
            Input(shape=input_shape),
            Conv2D(hidden_shape[0], kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(hidden_shape[1], kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


@ex.main
@ex.capture
def train_model(batch_size=128, epochs=5):
    x_train, y_train, x_test, y_test = load_data()

    model = basic_cnn()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return {"score": score}


if __name__ == '__main__':
    ex.run()
    # go to : localhost:9000

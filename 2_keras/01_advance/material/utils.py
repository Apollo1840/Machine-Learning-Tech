import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def save_keras_model(model, model_path):
    """
    smaller space consumption than model.save()

    :param model: keras trained model
    :param model_path: str
    :return:
    """

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # name convention
    model_architecture = "/model_architecture.json"
    model_weights = "/model_weights.h5"

    json_string = model.to_json()
    with open(model_path + model_architecture, "w") as f:
        json.dump(json_string, f)

    model.save_weights(model_path + model_weights)


def load_keras_model(model_path):
    """

    :param model_path:
    :return:
    """

    # name convention
    model_architecture = "/model_architecture.json"
    model_weights = "/model_weights.h5"

    with open(model_path + model_architecture, "r") as f:
        json_obj = json.load(f)

    model = model_from_json(json_obj)
    model.load_weights(model_path + model_weights)
    return model


def basic_callbacks(model_path):
    """
    how to use: model.fit(callbacks=basic_callbacks(model_path))

    :param model_path: str
    """
    return [ModelCheckpoint(model_path + '_ckpt.h5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='auto'),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
                               baseline=None, restore_best_weights=False),
            ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=0, mode='auto')]


if __name__ == "__main__":

    # train_test_split example
    X, y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Scaler example
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(X.reshape(-1, 1))
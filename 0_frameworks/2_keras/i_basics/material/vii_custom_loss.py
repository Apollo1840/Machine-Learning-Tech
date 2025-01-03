import keras.backend as K


def custom_mse(y_true, y_pred):
    """

    > model.compile(loss=custom_mse, optimizer='adam')

    :param: y_true: tensor(batch_size, ...)
    :param: y_pred: tensor(batch_size, ...)
    :return tensor(batch_size,)
    """

    # calculating squared difference between target and predicted values
    loss = K.square(y_pred - y_true)  # (batch_size, 2)

    # multiplying the values with weights along batch dimension
    loss = loss * [0.3, 0.7]  # (batch_size, 2)

    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)  # (batch_size,)

    return loss



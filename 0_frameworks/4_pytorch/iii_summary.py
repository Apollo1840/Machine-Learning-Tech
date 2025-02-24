from tqdm import tqdm

import torch
from torchsummary import summary
# pip install torchsummary

from material.data import fmnist_datagen, fmnist_classes
from ii_nn_gpu import NeuralNetwork, PytorchNN


class PytorchNNPro(PytorchNN):
    # transform the pytorch model to something similar to Keras model

    def summary(self):
        print(self)
        input_shape = (28, 28)
        summary(self, input_shape)

    def save(self, filename):
        """
        When saving model, it is recommended to save the model weights only for better flexibility.

        you can also save/load the optimizer status dict for continuing the training.
        """

        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename):
        model = cls()
        return model.load_state_dict(torch.load(filename))


if __name__ == '__main__':

    train_gen, test_gen = fmnist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()

    # summary
    model.summary()

    # use
    pred = model.predict(x_sample)
    print('Predicted: {}, Actual: {}'.format(
        fmnist_classes[pred[0].argmax(0)], fmnist_classes[y_sample]
    ))

    # save
    model.save("models/model.pth")
    print("Saved PyTorch Model State to model.pth")

    # load
    model = PytorchNN.load("models/model.pth")
    # model = PytorchNN()
    # model.load_state_dict(torch.load("model.pth"))

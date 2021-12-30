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
        input_shape = (3, 224, 224)
        summary(self, input_shape)

    def predict(self, x):
        self.eval()

        x = x.to("gpu")
        with torch.no_grad():
            y_pred = self(x)

        return y_pred

    def save(self, filename):
        torch.save(self.state_dict(), filename)


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
    model = PytorchNN()
    model.load_state_dict(torch.load("model.pth"))

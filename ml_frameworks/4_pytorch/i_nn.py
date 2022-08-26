from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from material.data import fmnist_datagen


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
        self.softmax = F.log_softmax

        # loss and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)  # call by reference

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        y = self.softmax(logits, dim=1)
        return y


class PytorchNN(NeuralNetwork):
    # transform the pytorch model to something similar to Keras model

    def fit_datagen(self, datagen, epochs=12):
        self.train()  # switch to train mode

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            for x, y in tqdm(datagen, total=len(datagen)):
                self.backprop(x, y)

            self.eval_datagen(datagen)

        print("Done!")

    def eval_datagen(self, datagen):
        # eval model
        self.eval()  # switch to evaluation mode

        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in datagen:
                y_pred = self(X)
                test_loss += self.loss(y_pred, y).item()
                correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        size = len(datagen.dataset)
        test_loss_avg = test_loss / size  # avg_loss
        acc = correct / size  # accuracy
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss_avg:>8f} \n")

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            y_pred = self(x)

        return y_pred

    # helper functions
    def backprop(self, x, y):
        """
        Backpropagation

        works as .step_function() in tensorflow.

        :param: x: x_batch
        :param: y: y_batch
        """

        # 1) clear gradient:
        self.optimizer.zero_grad()

        # 2) calculate the gradient:
        y_pred = self(x)
        self.loss(y_pred, y).backward()
        # regularly (y_pred, y) are called: (outputs, labels)

        # 3) update the paramters:
        self.optimizer.step()


if __name__ == '__main__':

    train_gen, test_gen = fmnist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()

    # train
    model.fit_datagen(train_gen, epochs=12)

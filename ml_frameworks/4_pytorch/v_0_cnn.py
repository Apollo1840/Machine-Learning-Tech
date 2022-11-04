from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from material.data import mnist_datagen


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # loss and optimizer
        self.loss = F.nll_loss
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=1e-3)  # call by reference

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class PytorchNN(CNN):

    def fit_datagen(self, datagen, epochs=12, verbose=False):
        self.train()

        loss_values = []
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (x, y) in tqdm(enumerate(datagen), total=len(datagen)):
                self.optimizer.zero_grad()

                y_pred = self(x)
                self.loss(y_pred, y).backward()

                self.optimizer.step()

                running_loss += self.loss(y_pred, y).item() * len(x)

            loss_value = running_loss/len(datagen)
            loss_values.append(loss_values)

            if verbose:
                print("Epoch {}/{}: loss: {}".format(epoch, epochs, loss_value))

        return {"history": {"loss": loss_values}}

    def eval_datagen(self, datagen):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in datagen:
                y_pred = self(x)
                test_loss += self.loss(y_pred, y, reduction='sum').item()  # sum up batch loss
                pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= len(datagen.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(datagen.dataset),
            100. * correct / len(datagen.dataset)))


def main():
    train_gen, test_gen = mnist_datagen()

    model = PytorchNN()

    n_epochs = 3
    for _ in range(1, n_epochs):
        model.fit_datagen(train_gen)
        model.eval_datagen(test_gen)


if __name__ == '__main__':
    main()

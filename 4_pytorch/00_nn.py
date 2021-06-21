import torch
from torch import nn

from material.data import fminist_datagen, fminist_classes


class NeuralNetwork(nn.Module):
    loss_fn = nn.CrossEntropyLoss()

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

        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)  # call by reference

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Pytorch_NN_simle(NeuralNetwork):

    def fit_datagen(self, datagen, epochs=12):

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            for X, y in datagen:
                loss = self.loss_fn(self(X), y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print("Done!")

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            pred = model(x)

        return pred


class PytorchNN(NeuralNetwork):

    def fit_datagen(self, datagen, epochs=12):

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            model.train()  # turn on the train mode
            for X, y in datagen:
                model.backprop(X, y)

            model.eval_datagen(test_gen)

        print("Done!")

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            pred = model(x)

        return pred

    def backprop(self, X, y):
        # Backpropagation
        self.optimizer.zero_grad()
        self.loss_fn(self(X), y).backward()
        self.optimizer.step()

    def eval_datagen(self, datagen):
        # eval model
        self.eval()  # change to evaluation mode

        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in datagen:
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size = len(datagen.dataset)
        test_loss_avg = test_loss / size  # avg_loss
        acc = correct / size  # accuracy
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss_avg:>8f} \n")


if __name__ == '__main__':

    # train a nn
    train_gen, test_gen = fminist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()

    # train
    model.fit_datagen(train_gen, epochs=12)

    # use
    pred = model.predict(x_sample)
    print('Predicted: {}, Actual: {}'.format(
        fminist_classes[pred[0].argmax(0)], fminist_classes[y_sample]
    ))

    # save
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # load
    model = PytorchNN()
    model.load_state_dict(torch.load("model.pth"))

import torch
from torch import nn

from material.data import fminist_datagen, fminist_classes


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class PytorchNN(CNN):

    def fit_datagen(self, datagen, epochs=12):

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            for X, y in datagen:
                model.backprop(X, y)

            model.eval_datagen(test_gen)

        print("Done!")

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            x = x.to("cuda")
            pred = model(x)

        return pred

    def backprop(self, X, y):
        X, y = X.to("cuda"), y.to("cuda")

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
                X, y = X.to("cuda"), y.to("cuda")

                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size = len(datagen.dataset)
        test_loss_avg = test_loss / size  # avg_loss
        acc = correct / size  # accuracy
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss_avg:>8f} \n")


if __name__ == '__main__':

    train_gen, test_gen = fminist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()
    model = model.to("cuda")

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
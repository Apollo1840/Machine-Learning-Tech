from tqdm import tqdm

import torch
from material.data import fmnist_datagen, fmnist_classes
from i_nn import NeuralNetwork


class PytorchNN(NeuralNetwork):
    # transform the pytorch model to something similar to Keras model

    def fit_datagen(self, datagen, epochs=12):
        self.train()

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            for x, y in tqdm(datagen, total=len(datagen)):
                self.backprop(x, y)

            self.eval_datagen(datagen)

        print("Done!")

    def predict(self, x):
        self.eval()

        x = x.to("cuda")
        with torch.no_grad():
            y_pred = self(x)

        return y_pred

    # helper functions
    def backprop(self, x, y):
        # Backpropagation
        self.optimizer.zero_grad()

        x, y = x.to("cuda"), y.to("cuda")
        y_pred = self(x)
        self.loss(y_pred, y).backward()
        # regularly they are called: (outputs, labels)

        self.optimizer.step()

    def eval_datagen(self, datagen):
        # eval model
        self.eval()  # change to evaluation mode

        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in datagen:
                x, y = x.to("cuda"), y.to("cuda")

                y_pred = self(x)
                test_loss += self.loss(y_pred, y).item()
                correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        size = len(datagen.dataset)
        test_loss_avg = test_loss / size  # avg_loss
        acc = correct / size  # accuracy
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss_avg:>8f} \n")


if __name__ == '__main__':

    train_gen, test_gen = fmnist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()
    model.to("cuda")

    # train
    model.fit_datagen(train_gen, epochs=12)

    # use
    pred = model.predict(x_sample)
    print('Predicted: {}, Actual: {}'.format(
        fmnist_classes[pred[0].argmax(0)], fmnist_classes[y_sample]
    ))

    # save
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # load
    model = PytorchNN()
    model.load_state_dict(torch.load("model.pth"))

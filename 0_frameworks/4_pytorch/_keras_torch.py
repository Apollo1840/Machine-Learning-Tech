from tqdm import tqdm

import torch
from torchsummary import summary
import torch.nn.functional as F


class KerasTorch():

    def __init__(self, torch_model):
        self.torch_model = torch_model

        # loss and optimizer
        self.loss = F.nll_loss
        self.optimizer = torch.optim.Adadelta(self.torch_model.parameters(), lr=1e-3)  # call by reference

    def fit_datagen(self, datagen):
        self.torch_model.train()
        for batch_idx, (x, y) in tqdm(enumerate(datagen), total=len(datagen)):
            self.optimizer.zero_grad()

            y_pred = self.torch_model(x)
            self.loss(y_pred, y).backward()

            self.optimizer.step()

    def eval_datagen(self, datagen):
        self.torch_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in datagen:
                y_pred = self.torch_model(x_batch)
                test_loss += self.loss(y_pred, y_batch, reduction='sum').item()  # sum up batch loss
                pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y_batch.view_as(pred)).sum().item()

        test_loss /= len(datagen.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(datagen.dataset),
            100. * correct / len(datagen.dataset)))

    def summary(self):
        print(self.torch_model)
        input_shape = (3, 224, 224)
        summary(self.torch_model, input_shape)

    def predict(self, x):
        self.torch_model.eval()

        x = x.to("gpu")
        with torch.no_grad():
            y_pred = self.torch_model(x)

        return y_pred

    def save(self, filename):
        # .pt file
        torch.save(self.torch_model, filename)

    @classmethod
    def load_model(cls, filename):
        # .pt file
        cls(torch.load(filename))



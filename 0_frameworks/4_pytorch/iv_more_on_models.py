from tqdm import tqdm

import torch
from torchsummary import summary
# pip install torchsummary

from material.data import fmnist_datagen, fmnist_classes
from ii_nn_gpu import NeuralNetwork, PytorchNN


if __name__ == '__main__':

    train_gen, test_gen = fmnist_datagen()
    for x, y in test_gen:
        x_sample, y_sample = x[0], y[0]
        break

    model = PytorchNN()

    print([module for module in model.children()])
    # print([module for module in model.modules()])  # recursively
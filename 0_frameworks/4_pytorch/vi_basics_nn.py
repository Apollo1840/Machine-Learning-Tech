import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


net = Net()
print(net)

# knowledge
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# data
input_x = torch.randn(1, 1, 32, 32)
out_y = net(input_x)
print(out_y)
target = torch.tensor(np.random.randn(1, 10), dtype=torch.float)

# train the model
# create your optimizer, connect Knowledge(parameters) to the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# before training in one batch
optimizer.zero_grad()  # zero the gradient buffers

loss = nn.MSELoss()(net(input_x), target)
loss.backward()  # give the parameters required ,grad (the delta w)

optimizer.step()  # Does the update, defines how parameters do with the .grad data.

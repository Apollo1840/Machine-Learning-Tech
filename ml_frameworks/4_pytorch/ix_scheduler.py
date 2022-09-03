"""
scheduler is 'like' a wrapper for optimizer,

you can simply wrap the optimizer, and use scheduler.step() after optimizer.step()


"""
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

model = torch.nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=100)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
lrs = []

for i in range(10):
    optimizer.step()
    scheduler.step()

    lrs.append(optimizer.param_groups[0]["lr"])
    #  print("Factor = ",0.1 if i!=0 and i%2!=0 else 1," , Learning Rate = ",optimizer.param_groups[0]["lr"])

plt.plot(range(10), lrs)

# if you want to use customized Scheduler, you could do:
# > for param_group in optimizer.param_groups:
# >     param_group["lr"] = ...


# -------------------------------------------------------------------------------------------------------
# for callbacks like eary stopping:

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                return True
        return False


early_stopping = EarlyStopping(tolerance=5, min_delta=10)

for i in range(1200):

    epoch_train_loss = 1.0

    with torch.no_grad():
        epoch_validate_loss = 1.5

    if early_stopping(epoch_train_loss, epoch_validate_loss):
        print("We are at epoch:", i)
        break
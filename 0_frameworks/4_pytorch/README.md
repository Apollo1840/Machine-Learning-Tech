# Pytorch

## Tutorial material
install: 
    
    pip install torch torchvision

## Learning trace

1. Learn how to build pytorch model and train/use it like keras (i). 
2. train with gpu (ii).
3. learn how to view, store and load model (iii).
4. detailed operations of pytorch model (iv).
5. more pytorch models. (v_*)
6. learn the basics and understand how does pytorch model is trained. (vi)
7. learn how to create custom layers. (vii)
8. learn how to create custom loss. (viii)
9. scheduler (callbacks in keras). (ix)
10. multiple GPU, device control (x)
11. learn tricks from others repo. (t_*)

(Suggestion: prefer to read `.ipynb` than `.py`.)

## tricks

### load_data
Normally we do not load all the data into RAM.
We use something like `datasets.ImageFolder` to store all the links(filepaths) of the data.
When initialize such LinksStore we also define our `torchvision.transforms` as data preprocessing steps, 
eg. resize, normalize. etw.

We use something like `DataLoader`, which is connected to the LinksStore as the actual data generator.
Inside the `DataLoader`, how to load the data is defined, as well as batch_size. 
And here comes the first optimization trick, 
we could use **num_workers** and **pin_memory** to speed up training process by speed up batch data loading happened in train step.

### auto-cast
Use autocast while computing the loss with increase speed.

```python
    with torch.cuda.amp.autocast():
        loss = ...
```


### larger batch_size when RAM is not enough.
The trick is very simple. Just zero your gradient NOT after each batch, but some batch.

(Note: The effect is NOT exact same to larger batch_size)

```python
    # increase batch_size 8x
    if (data_iter_step + 1) % 8 == 0:
        optimizer.zero_grad()

    #  if data_iter_step % 8 == 0:
    #       lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch)
    
    # loss_scaler(loss, optimizer, parameters, update_grad=(data_iter_step + 1) % accum_iter == 0)
```


### loss_scaler
This is a numerical trick to help computer to calculate loss and its gradient.

```python
    loss_scaler = NativeScaler()
    
    # loss = ...
    # model = ...
    # optimizer = ...
    loss_scaler(loss, optimizer, parameters=model.parameters())
    # loss.backward() and optimzer.step() is inside loss_scaler
```


## material
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

- examples： 
  - https://github.com/pytorch/examples/blob/master/mnist/main.py
  - https://github.com/pytorch/examples
- custom layer: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
- basic operations : https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/

# Pytorch-Lighting
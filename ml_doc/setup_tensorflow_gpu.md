# How to enable training the Tensorflow models with GPU

## Step 1: Install conda

ref: [How to Install Anaconda on Ubuntu 18.04 and 20.04](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04)


```bash

  cd /tmp
  curl –O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
  bash Anaconda3-2020.02-Linux-x86_64.sh
  
  # optional: check whether is it successfully installed
  source ~/.bashrc
  conda info

```

## Step 2: build a tensorflow virtual environment


```bash
    conda create --name tf_gpu tensorflow-gpu 
    
    # optional: activate the virtual environment
    conda activate tf_gpu
```

Now you can run your python script under this virtual environment.

Other related commands:
```bash
    # check python version
    python -V
    
    # check tensorflow version
    pip list
    
    # install packages
    pip install pandas
    
    # to deactivate the env
    deactivate
```

## Step 3: Check whether tensorflow uses GPU

Run `_test_gpu.ipynb`.

## (Optional) Step 4: add this conda env to jupyter notebook

```bash
    conda activate tf_gpu
    python -m ipykernel install --user --name=tf_gpu
    
    # optional: check whether is it successfully added to the kernels
    jupter notebook
```

## (Optional) Step 5: add this conda env to pycharm

Go to `file -> settings -> Project: -> python interpreter -> (the gear sign after the env) -> add`.

Add this existed env:  `~/anaconda3/envs/tf_gpu/bin/python3`


# Alternative

The traditional way with `Nvidia driver + cuda + cuDNN + Tensorflow-gpu` reference: 

https://wandb.ai/wandb/common-ml-errors/reports/How-to-Correctly-Install-TensorFlow-in-a-GPU-Enabled-Laptop---VmlldzozMDYxMDQ


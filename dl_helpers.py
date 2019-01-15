# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:29:49 2018

@author: zouco
"""
import numpy as np

#--------------------------------------------------------------------
# create mini-batch
class Batch(object):
    '''
        how to use:  
                batch = Batch(x_data,y_data,80)
                x_data, y_data = batch.nextBatch()
                # or
                x_data, y_data = batch.nextBatch_enum()  # this will enumerate all data entry
    
    '''
    
    
    def __init__(self, X, y, batch_size):
        # X, y should be columnwise data
        
        self.X = X
        self.y = y
        assert(self.X.shape[0] == self.y.shape[0])
        
        self.batch_size = batch_size
        self.remain_entry = list(range(self.y.shape[0]))
    
    def getBatch(self):
        indices = np.random.choice(range(self.y.shape[0]), self.batch_size)
        return self.X[indices, :], self.y[indices, :]
    
    def getBatch_enum(self):
        if len(self.remain_entry) == 0:
            return []
        
        if len(self.remain_entry) < self.batch_size:
            indices = []
            indices.extend(self.remain_entry)
        else:    
            indices = np.random.choice(self.remain_entry, self.batch_size, replace=False)
        
        if indices is not None:
            for i in indices:
                self.remain_entry.remove(i)
        return self.X[indices, :], self.y[indices, :]
    

batch = Batch(x_data,y_data,80)
print(x_data.shape)
x_b,y_b = batch.getBatch_enum()
print(x_b.shape)
x_b,y_b = batch.getBatch_enum()
print(x_b.shape)
x_b,y_b = batch.getBatch_enum()
print(x_b.shape)

print(x_b)
print(y_b)
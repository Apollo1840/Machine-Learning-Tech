# -*- coding: utf-8 -*-
"""
in this py file, I will show how to use 
    regularization term, 
    dropout method
    Batch norm method
    
and a better initialization of weights to easy the gradient vanishing problem

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# data
# if data is less, NN will work extremely bad
# also, y should not be too big better between 0 and 1

x = np.linspace(-10,10,num=200)[:,None]   # [:,None] make the x.shape to be (30,1)
y = -0.1*x + 0.2*x**2 + 0.3*x**3 + 20*np.random.randn(200,1)
plt.plot(x,y,'o')
plt.show()

# data preprocessing

from sklearn.model_selection import train_test_split
x_data = x
y_data = y
x_train, x_test, y_train, y_test = train_test_split(x,y)


# simple nn with [1，10,5,1]
# the dimension is very important issue of NN, here we use column-wise data as training data

x_dim = x_train.shape[1]
y_dim = y_train.shape[1]

x = tf.placeholder(tf.float32, [x_dim, None])
y = tf.placeholder(tf.float32, [y_dim, None])


n = [x_dim]
w = [None]
b = [None]
z = [None]
a = [x]    

    
p=1
n.append(20)
w.append(tf.Variable(tf.random_normal([n[p],n[p-1]])))
b.append(tf.Variable(tf.zeros([n[p],1])))
z.append(tf.matmul(w[p],a[p-1])+b[p])
a.append(tf.nn.tanh(z[p]))
# a.append(tf.nn.dropout(tf.nn.tanh(z[p]), keep_prob = 0.5))

p=p+1
n.append(5)
# w.append(tf.Variable(tf.random_normal([n[p],n[p-1]])))
# better initialization:
w.append(tf.Variable(tf.random_normal([n[p],n[p-1]])*(tf.sqrt(1/n[p-1]))))

b.append(tf.Variable(tf.zeros([n[p],1])))
z.append(tf.matmul(w[p],a[p-1])+b[p])
a.append(tf.nn.tanh(z[p]))
# a.append(tf.nn.dropout(tf.nn.tanh(z[p]), keep_prob = 0.8))

p=p+1
n.append(y_dim)
w.append(tf.Variable(tf.random_normal([n[p],n[p-1]])))
b.append(tf.Variable(tf.zeros([n[p],1])))
z.append(tf.matmul(w[p],a[p-1])+b[p])

# # batch normalization
# scale = tf.Variable(tf.ones([n[p],1]))
# shift = tf.Variable(tf.zeros([n[p],1]))
# fc_mean, fc_var = tf.nn.moments(z[p], axis = 1)
# z[p] = tf.nn.batch_normalization(z[p], z_mean, z_var, shift, scale, episilon=10e-5)

a.append(z[p])

loss = tf.reduce_mean(tf.square(a[p]-y)/10**3)

# regularization
beta = 10**(-4)
for i in range(len(w)):
    if i>0:
        loss += beta*tf.nn.l2_loss(w[i])


optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={x: x_train.T, y: y_train.T})
        if i%100 == 0:
            print(sess.run(loss, feed_dict={x: x_train.T, y: y_train.T}))
    y_predict = sess.run(a[p], feed_dict={x: x_data.T})
    loss_train = sess.run(loss, feed_dict={x: x_train.T, y: y_train.T})
    loss_test = sess.run(loss, feed_dict={x: x_test.T, y: y_test.T})

plt.scatter(x_data, y_data)
rl=sorted(list(zip(x_data.ravel(),y_predict.ravel())))
plt.plot([i for i,j in rl],[j for i,j in rl], 'r-', lw=5)
plt.show()

print(loss_train)
print(loss_test)



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



# Norm batch




# -*- coding: utf-8 -*-

'''
the original code is from https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
thanks to R2RT and his work. It is the best implementation of RNN tutorial I have ever seen.
this can be only used to educational purpose, if you want to use it in other situations, please contact R2RT

'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
size=10
X = np.array(np.random.choice(2, size=(size,)))
print(X)


# not used      
def list_block(a,i,len_block=1):
    return a[len_block * (i-1):len_block * i]


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    batch_partition_length = len(raw_x) // batch_size
    
    # initialize
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    
    # further divide batch partitions into num_steps for truncated backprop
    seq_size = batch_partition_length // num_steps

    for i in range(seq_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)
        
# now, you can acess the data by:
#   for epoch in gen_epochs(n_epochs, num_steps):
#       for (X,Y) in epoch:
        



# Turn our x placeholder into a list of one-hot tensors:
# rnn_inputs is a list of num_steps tensors with shape [batch_size, num_classes]
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.float32, [batch_size, num_steps], name='labels_placeholder')
num_classes = 2
    
x_one_hot = tf.one_hot(x, num_classes)
# [1,0,1],[1,0,0] ti [0,1][1,0][0,1], [..][..][..]

rnn_inputs = tf.unstack(x_one_hot, axis=1)
# I want rnn_input for rnn_inputs, so the second dimension should come first



"""
see the tf.name_scope and tf.variable_scope


with tf.variable_scope('rnn_cell'):
    # W = Wxa + Waa
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    
    # return the a (the state)
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

"""

state_size = 4

with tf.variable_scope('rnn_cell'):
    # W = Wxa + Waa
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

init_state = tf.zeros([batch_size, state_size])
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    
    with tf.variable_scope('rnn_cell', reuse=True):
        # W = Wxa + Waa
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    
    state = tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]


with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
#          logit, label in zip(logits, y_as_list)]

y_pred = tf.transpose(predictions,[2,1,0])
losses = -y*tf.log(y_pred[0])

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.1).minimize(total_loss)
"""
Train the network
"""

def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses
training_losses = train_network(1,num_steps)
plt.plot(training_losses)
        
tf.reset_default_graph()




#########################################################################
# to undestand and use this, keep in mind:
# what is rnn_inputs
# what is init_state
# what is outputs
# what is final state

cells = tf.nn.rnn_cell.BasicRNNcell(state_size)
rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cells,rnn_inputs,initial_state=init_state)











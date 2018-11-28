# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 0. how to build a simply NN:

# assume we already have data, and we ignore the feed for the train
# so the following programming can not work without some concret cases

x_train =
y_train =

# assume the x_train is (1000, 5) data , y_train is (1000,1)
x = tf.placeholder(tf.float32, [None, 5])
y = tf.placeholder(tf.float32, [None, 1])

n1 = 4
w1 = tf.Variable(tf.random_normal([5, n1]))  # it is transpose of the form in textbook of NN
b1 = tf.Variable(tf.zeros([1, n1])) # b1 is (1,4) because the output is 1 dimensional
z1 = tf.matmul(x, w1) + b1 # here broadcast will happen
a1 = tf.nn.relu(z1)

n2 = 1
w2 = tf.Variable(tf.random_normal([n1, 1]))
b2 = tf.Variable(tf.zeros([1, n2]))
z2 = tf.matmul(a1,w1) + b2
a2 = z2

loss = tf.reduce_mean(tf.square(a2-y))

optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.initializers.global_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        sess.run(train, feed_dict={x: x_train, y: y_train})
    print(sess.run(loss, feed_dict={x: x_train, y: y_train}))


# -----------------------------------------------------------------------------
# 1. how to build a real NN:
# here we use mini-batch, and we consider adjusting learning rate

# minst
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size
# we need n_batch to write the loop 
# in the loop we will use: 
# batch_xs, batch_ys = mnist.train.next_batch(batch_size)


# new we going to build the NN; lets start with placeholder for data and parameter
pic             = tf.placeholder(tf.float32, [None, 784])
the_label       = tf.placeholder(tf.float32, [None, 10])
learning_rate   = tf.placeholder(tf.float32) 

# layer 1
nb_n_1 = 500
w1 = tf.Variable(tf.truncated_normal([784, nb_n_1], stddev = 0.1))
b1 = tf.Variable(tf.zeros([nb_n_1]) + 0.1)
l1 = tf.nn.tanh(tf.matmul(pic, w1) + b1)
nb_n = nb_n_1
   
# layer 2
nb_n_2 = 300
w2 = tf.Variable(tf.truncated_normal([nb_n, nb_n_2], stddev = 0.1))
b2 = tf.Variable(tf.zeros([nb_n_2]) + 0.1)
l2 = tf.nn.tanh(tf.matmul(l1, w2) + b2)
nb_n = nb_n_2
   
# layer 3
nb_n_3 = 10
w3 = tf.Variable(tf.truncated_normal([nb_n, 10], stddev = 0.1))
b3 = tf.Variable(tf.zeros([nb_n_3]) + 0.1)
l3 = tf.nn.softmax(tf.matmul(l2, w3) + b3) # softmax layer takes no activation function

loss = tf.losses.softmax_cross_entropy(the_label, l3)
# loss = tf.reduce_mean(tf.square(the_label-l3))  

optimizer =  tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss) 

# ww also want to see some performance during the training
correct_predict = tf.equal(tf.argmax(l3, 1), tf.argmax(the_label, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
# tf.cast make True to be 1

with tf.Session() as sess:
   sess.run(tf.initializers.global_variables())
   n_epoch = 200
   for epoch in range(n_epoch + 1):
       for batch in range(n_batch):
           batch_xs, batch_ys = mnist.train.next_batch(batch_size)
           sess.run(train, feed_dict={pic:batch_xs, the_label:batch_ys, learning_rate: 0.001 * (0.98**epoch)})
       if epoch % 2 == 0:
           acc = sess.run(accuracy, feed_dict={pic: mnist.test.images, the_label: mnist.test.labels})
           print('{}%\tIteration {} : accuracy : {}'.format(float(epoch)*100/n_epoch, epoch, acc))

tf.reset_default_graph()






# -----------------------------------------------------------------------------
# 2. how to build it with tensorboard
# First we need to carefully arrange our namescope
# Then we put tf.summary.scalar or tf.summary.histogram where we need
# Then we merge the summaries by: tf.summary.merge_all()
# finally, during the sess, we sess.run the mergerd and write the output (smy) to writer:
# how? by 
# writer = tf.summary.FileWriter('logs/', sess.graph) 
# and
# writer.add_summary(smy, epoch)


import tensorflow as tf


def variable_summaries(var):
    with tf.name_scope('summarises'):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
            
            

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 98% minist    
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
    
n_dim = 784
n_out = 10
        
with tf.name_scope('input'):
    with tf.name_scope('pic'):
        x = tf.placeholder(tf.float32, [None, n_dim])
        
    with tf.name_scope('label'):
        y = tf.placeholder(tf.float32, [None, n_out])
        
# layer 1
with tf.name_scope('layer_1'):
    nb_n_1 = 500
            
    with tf.name_scope('weights'):  # inside namescope, the variable will get its appended name
        w1 = tf.Variable(tf.truncated_normal([n_dim, nb_n_1], stddev = 0.1))
        variable_summaries(w1)
            
    with tf.name_scope('bias'):
        b1 = tf.Variable(tf.zeros([nb_n_1]) + 0.1)
        variable_summaries(b1)
            
    a1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    nb_n = nb_n_1
       
# layer 2
with tf.name_scope('layer_2'):
    nb_n_2 = 300
    
    with tf.name_scope('weights'):
        w2 = tf.Variable(tf.truncated_normal([nb_n, nb_n_2], stddev = 0.1))
        variable_summaries(w2)
    
    with tf.name_scope('bias'):
        b2 = tf.Variable(tf.zeros([nb_n_2]) + 0.1)
        variable_summaries(b2)
    
    a2 = tf.nn.tanh(tf.matmul(a1, w2) + b2)
    nb_n = nb_n_2
       
# layer 3
with tf.name_scope('layer_3'):
    nb_n_3 = n_out
    
    with tf.name_scope('weights'):
        w3 = tf.Variable(tf.truncated_normal([nb_n, nb_n_3], stddev = 0.1))
        variable_summaries(w3)
    
    with tf.name_scope('bias'):
        b3 = tf.Variable(tf.zeros([nb_n_3]) + 0.1)
        variable_summaries(b3)
        
    a3 = tf.nn.softmax(tf.matmul(a2, w3) + b3)

# model & train
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-a3))
    # better : loss = -tf.reduce_mean(y * tf.log(a3)) * 1000.0
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizier'):
    learning_rate =  tf.placeholder(tf.float32) 
    optimizer =  tf.train.AdamOptimizer(learning_rate)

with tf.name_scope('train'):
    train = optimizer.minimize(loss) 
        

# information
with tf.name_scope('accuracy'):
    correct_predict = tf.equal(tf.argmax(a3, 1), tf.argmax(y, 1)) 
    # argmax is biggest location
    # this is a list of bool
    
    # tf.cast make True to be 1
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    
    tf.summary.scalar('accuracy', accuracy)
   

init = tf.initialize_all_variables()
merged = tf.summary.merge_all()
    
    
with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph) 
    sess.run(init)
    n_epoch = 50
    for epoch in range(n_epoch + 1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #  sess.run(train, feed_dict={x:batch_xs, y:batch_ys, learning_rate: 0.001 * (0.98**epoch)})
            #  smy = sess.run(merged, feed_dict = {x:batch_xs, y:batch_ys, learning_rate: 0.001 * (0.98**epoch)})
            
            _,smy = sess.run([train, merged], feed_dict={x:batch_xs, y:batch_ys, learning_rate: 0.001 * (0.98**epoch)})
            writer.add_summary(smy, epoch)
            
        if epoch % 2 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('{}%\tIteration {} : accuracy : {}'.format(float(epoch)*100/n_epoch, epoch, acc))
    
tf.reset_default_graph()

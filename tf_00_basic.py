# -*- coding: utf-8 -*-

# tensorflow work in a session
# it has objects like: 
#   tensor (data, normally use placeholder), 
#   variable (the weights), 
#   operator (construct loss function)
#   optimizer (when use minimize function, it output a operator)


# 0. how tf works
import tensorflow as tf
import numpy as np

data = np.array([[1,2,3],[4,5,6]])

x1 = tf.placeholder(tf.float32, [1,3])
x2 = tf.placeholder(tf.float32, [3,1]) 
product = tf.matmul(x1, x2)

with tf.Session() as sess:
    print(sess.run(product, feed_dict={x1:data[0].reshape(1,3), x2:data[1].reshape(3,1)}))
# note: reshape is necessary because (3,) can not work



# 1, how to use tf to do optimization
data = np.array([2,2]).reshape(1,-1)

x = tf.placeholder(tf.float32, [1,None]) # None means dont know
w = tf.Variable(tf.random_normal([1,2]))
cost = tf.matmul(x-w, tf.transpose((x-w), [1,0])) 
optimizer = tf.train.AdamOptimizer(0.1)

train = optimizer.minimize(cost)


with tf.Session() as sess:
    # since we use Variable we need to initialize it
    sess.run(tf.global_variables_initializer())  
    
    for _ in range(100):
        sess.run(train, feed_dict={x:data})
        print(sess.run(cost, feed_dict={x:data}))
    print('\nw suppose to be [2,2], we got: ', sess.run(w))










# bonus: 
c1 = tf.constant(1)
c2 = tf.constant(2)
c3 = tf.constant(3)

p1 = tf.placeholder(tf.float32)  # tensor
p2 = tf.placeholder(tf.float32)
a=1
b=2
#  pic = tf.placeholder(tf.float32, [None, n_dim])

# m1 = tf.constant([[3,3]])
# m2 = tf.constant([[2],[3]])
    
# w1 = tf.Variable(tf.random_normal([n_dim, nb_n_1]))
# update = tf.assign(state, tf.add(state, 1))  # only variable can be assigned. Constant can not be assigned, tensor cannot been assigned

add = tf.add(c1, c2)
ma = tf.multiply(add, c3)
mul = tf.multiply(p1, p2)

# fetch and feed

with tf.Session() as sess:
    # fetch
    print(sess.run([add, ma]))
    
    # feed
    print(sess.run(mul, feed_dict={p1: [a], p2: [b]}))
    





# -*- coding: utf-8 -*-

import tensorflow as tf

# part 1

# one_hot
oh0 = tf.one_hot([0,1,2,3,4], depth=10)
oh1 = tf.one_hot([0,1,2,3,4], depth=3)
oh2 = tf.one_hot([[0,1,1],[1,0,1],[1,0,0]], depth=2)
  
# reshape 
x = [0., 1., 2., 3., 4., 5., 6., 7.]
y = tf.reshape(x, [2, 2, 2])

# unstack 
# unstack is rolling the index to the first: e.g when axis =1 : [1,2,3] -> [2,3,1] 
y0 = tf.unstack(y, axis=0)
y1 = tf.unstack(y, axis=1)
y2 = tf.unstack(y, axis=2)

# transpose
y0_2 = tf.transpose(y, [0,1,2])
y1_2 = tf.transpose(y, [1,0,2])
y2_2 = tf.transpose(y, [2,0,1])


    
with tf.Session() as sess:
    print('---------------one hot---------------------')
    print('---------\n oh0 \n')
    print(sess.run(oh0))
    
    print('---------\n oh1 \n')
    print(sess.run(oh1))
    
    print('---------\n oh2 \n')
    print(sess.run(oh2))
    
    print('---------------unstack---------------------')
    print('---------\n y \n')
    print(sess.run(y))
    print('---------\n y0 \n')
    print(sess.run(y0))
    print('---------\n y1 \n')
    print(sess.run(y1))
    print('---------\n y2 \n')
    print(sess.run(y2))
    
    print('---------------transpose---------------------')
    print('---------\n y0_2 \n')
    print(sess.run(y0_2))
    print('---------\n y1_2 \n')
    print(sess.run(y1_2))
    print('---------\n y2_2 \n')
    print(sess.run(y2_2))


#-----------------------------------------------------------------------------
# part 2
import numpy as np
a = np.array([[1,2,3,4,5]])
a2 = tf.squeeze(a)

b = np.array([[[1],[2]],[[3],[4]]])
b2 = tf.squeeze(b)

    
with tf.Session() as sess:
    print('---------------squeeze---------------------')
    print('-----------\n a2 \n')
    print(a)
    print(a.shape)
    a2_ = sess.run(a2)
    print('\nsqueezed\n:')
    print(a2_)
    print(a2_.shape)
    
    print('-----------\n b2 \n')
    print(b)
    print(b.shape)
    b2_ = sess.run(b2)
    print('\nsqueezed\n:')
    print(b2_)
    print(b2_.shape)
    
    
    


# show structure

data = tf.placeholder("float", shape=[1, 7, 7, 1])
conv = tf.nn.conv2d(data, filter=tf.Variable(tf.constant(0.1,shape=[4, 4, 1, 1])), strides=[1, 2, 2, 1], padding='SAME')

print(conv.get_shape())  # prints (1, 4, 4, 1), but should be (1, 5, 5, 1)




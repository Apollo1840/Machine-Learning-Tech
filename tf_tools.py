# -*- coding: utf-8 -*-

import tensorflow as tf


# tutorial about tf.one_hot and tf.unstack:
oh0 = tf.one_hot([0,1,2,3], depth=3)
oh1 = tf.one_hot([[0,1,1],[1,0,1],[1,0,0]], depth=2)
   
x = [0., 1., 2., 3., 4., 5., 6., 7.]
y = tf.reshape(x, [2, 2, 2])

# unstack is rolling the index to the first: e.g when axis =1 : [1,2,3] -> [2,3,1] 
y0 = tf.unstack(y, axis=0)
y1 = tf.unstack(y, axis=1)
y2 = tf.unstack(y, axis=2)

y0_2 = tf.transpose(y, [0,1,2])
y1_2 = tf.transpose(y, [1,0,2])
y2_2 = tf.transpose(y, [2,0,1])


    
with tf.Session() as sess:
    # print(sess.run(oh0))
    
    # x = sess.run(oh1)
    # print(x)
    # print(x[0])
    # print(x[0][0])
    
    print(sess.run(y))
    print()
    print(sess.run(y0))
    print()
    print(sess.run(y1))
    print()
    print(sess.run(y2))
    print()
    print(sess.run(y0_2))
    print()
    print(sess.run(y1_2))
    print()
    print(sess.run(y2_2))


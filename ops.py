# coding = utf-8
'''
Created on Apr 2 15:38:37 2019
@author: huyz
'''

import tensorflow as tf

def conv2d(x, filters=64, kernel_size=3, strides=1, padding='SAME', scope='conv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        return x

def max_pool(x, pool_size=2, strides=2, padding='SAME', scope='max_pool'):
    with tf.variable_scope(scope):
        x = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides, padding=padding)
        return x

def avg_pool(x, pool_size=5, strides=3, padding='SAME', scope='avg_pool'):
    with tf.variable_scope(scope):
        x = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=3, padding=padding)
        return x

def fully_connected(x, units, name='fc'):
    return tf.layers.dense(x, units=units, name=name)

def relu(x):
    return tf.nn.relu(x)

def flatten(x):
    return tf.layers.flatten(x)

def lrn(x, name='lrn'):
    return tf.nn.local_response_normalization(input=x,
                                              depth_radius=2,
                                              alpha=2e-05,
                                              beta=0.75,
                                              name=name)

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
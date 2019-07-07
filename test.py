# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:22:13 2019

@author: huyz
"""
import tensorflow as tf
import os
import numpy as np
from ops import *


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
#init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "wo":init_weights([4096,64])
    }

#init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "bo":init_weights([64])
    }

def conv2d(x,w,b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def pooling(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

def norm(x,lsize = 4):
    return tf.nn.lrn(x,depth_radius = lsize,bias = 1,alpha = 0.001/9.0,beta = 0.75)


def mmodel(images):
    l1 = conv2d(images,weights["w1"],biases["b1"])
    l2 = pooling(l1)
    l2 = norm(l2)
    l3 = conv2d(l2,weights["w2"],biases["b2"])
    l4 = pooling(l3)
    l4 = norm(l4)
    l5 = conv2d(l4,weights["w3"],biases["b3"])
    #same as the batch size
    l6 = pooling(l5)
    # l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
    l6 = tf.reshape(l6, [-1, l6.get_shape().as_list()[1]*l6.get_shape().as_list()[2]*l6.get_shape().as_list()[3]])
    l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
    soft_max = tf.add(tf.matmul(l7,weights["wo"]),biases["bo"])
    return soft_max

import pandas as pd

IMAGE_SIZE = 32
BATCH_SIZE = 16

def _parse_fn(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_normlized = tf.cast(image_decoded, tf.float32) / 127.5 - 1
    image_resized = tf.image.resize_images(image_normlized, (IMAGE_SIZE, IMAGE_SIZE))
    
    return image_resized, label

train_csv = pd.read_csv('E:/data/mini-imagenet/mini-imagenet/train.csv')
train_filenames = tf.constant(['E:/data/mini-imagenet/mini-imagenet/images/' + fname for fname in train_csv['filename'].tolist()])
train_labels = tf.constant(train_csv['label'].tolist())

train_data = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_data = train_data.map(_parse_fn)
train_data = train_data.shuffle(buffer_size=1000).batch(BATCH_SIZE).repeat(10)
# batch_images, batch_labels = train_data.make_one_shot_iterator().get_next()
iterator = train_data.make_one_shot_iterator()
one_batch = iterator.get_next()


def loss(logits,label_batches):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
    cost = tf.reduce_mean(cross_entropy)
    return cost

def get_accuracy(logits,labels):
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc

def training(loss,lr):
    train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    return train_op

def run_training():
    sess = tf.Session()
    batch_images, batch_labels = sess.run(one_batch)
    p = mmodel(batch_images)
    cost = loss(p,batch_labels)
    train_op = training(cost,0.001)
    acc = get_accuracy(p,batch_labels)
    
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(2000):
        _, a, l = sess.run([train_op, acc, cost])
        print('Step: %3d, Loss: %.6f, Accuracy: %.6f'%(i+1, l, a))
    
if __name__ == '__main__':
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_images, batch_labels = sess.run(one_batch)
    print(batch_labels)
#    print(batch_images.shape)
    print(batch_labels.shape)
    p = mmodel(batch_images)
    print(p.shape)
    print(sess.run(p))
    # run_training()
    # print(batch_images, batch_labels)





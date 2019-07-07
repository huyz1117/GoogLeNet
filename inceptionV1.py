# coding = utf-8
'''
Created on Apr 1 22:01:25 2019
@author: huyz
'''

import os
import numpy as np
import argparse
import tensorflow as tf

from ops import *

def inception(net, filters, scope='inception'):
    with tf.variable_scope(scope):
        '''
        Arguments:
        
        f1: inception模块最左边分支1x1卷积输出通道数
        f3_r: inception模块3x3卷积之前经1x1降维的通道数
        f3: inception模块3x3卷积输出通道数
        f5_r: inception模块5x5卷积之前经1x1降维的通道数
        f5: inception模块5x5卷积输出通道数
        '''
        f1, f3_r, f3, f5_r, f5, f_pool = filters
        
        conv1 = conv2d(net, f1, 1, 1, scope='1x1_conv')
        
        conv3_r = conv2d(net, f3_r, 1, 1, scope='3x3_conv_r')
        conv3 = conv2d(conv3_r, f3, 3, 1, scope='3x3_conv')
        
        conv5_r = conv2d(net, f5_r, 1, 1, scope='5x5_conv_r')
        conv5 = conv2d(conv5_r, f5, 5, 1, scope='5x5_conv')
        
        pool = max_pool(net, 3, 1, scope='max_pool_r')
        pool_conv = conv2d(pool, f_pool, 1, 1, scope='max_pool_conv')
        
        out = tf.concat([conv1, conv3, conv5, pool_conv], axis=-1)
        
        return out

def inception_v1(x, dropout, reuse, is_training, scope='inception_v1'):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv2d(x, 64, 7, 2, scope='conv1')
        conv1 = relu(conv1)
        pool1 = max_pool(conv1, 3, 2, scope='max_pool1')
        pool1 = lrn(pool1)
        
        conv2 = conv2d(pool1, 192, 3, 1, scope='conv2')
        conv2 = relu(conv2)
        pool2 = max_pool(conv2, 3, 2, scope='max_pool2')
        pool2 = lrn(pool2)
        
        inception_3a = inception(pool2, [64, 96, 128, 16, 32, 32], scope='inception_3a')
        inception_3b = inception(inception_3a, [128, 128, 192, 32, 96, 64], scope='inception_3b')
        
        pool3 = max_pool(inception_3b, 3, 2, scope='max_pool3')
        
        inception_4a = inception(pool3, [192, 96, 208, 16, 48, 64], scope='inception_4a')
        
        '''
        辅助分类器1
        '''
        ax1_pool = avg_pool(inception_4a, 5, 3, padding='VALID', scope='ax1_pool')
        ax1_conv = conv2d(ax1_pool, 128, 1, 1, scope='ax1_conv')
        ax1_conv = relu(ax1_conv)
        ax1_conv = tf.layers.dropout(ax1_conv, rate=dropout, training=is_training)
        ax1_flatten = flatten(ax1_conv)
        ax1_fc1 = fully_connected(ax1_flatten, 1024, name='ax1_fc1')
        ax1_out = fully_connected(ax1_fc1, 100, name='ax1_out')
        
        inception_4b = inception(inception_4a, [160, 112, 224, 24, 64, 64], scope='inception_4b')
        inception_4c = inception(inception_4b, [128, 128, 256, 24, 64, 64], scope='inception_4c')
        inception_4d = inception(inception_4c, [112, 144, 288, 32, 64, 64], scope='inception_4d')
        inception_4e = inception(inception_4d, [256, 160, 320, 32, 128, 128], scope='inception_4e')
        
        '''
        辅助分类器2
        '''
        ax2_pool = avg_pool(inception_4e, 5, 3, padding='VALID', scope='ax2_pool')
        ax2_conv = conv2d(ax2_pool, 128, 1, 1, scope='ax2_conv')
        ax2_conv = relu(ax2_conv)
        # ax2 = tf.nn.dropout(ax2_conv, keep_prob=0.3)
        ax2 = tf.layers.dropout(ax2_conv, rate=dropout, training=is_training)

        ax2_flatten = flatten(ax2_conv)
        ax2_fc1 = fully_connected(ax2_flatten, 1024, name='ax2_fc1')
        ax2_out = fully_connected(ax2_fc1, 100, name='ax2_out')
        
        pool4 = max_pool(inception_4e, 3, 2, scope='max_pool4')
        
        inception_5a = inception(pool4, [256, 160, 320, 32, 128, 128], scope='inception_5a')
        inception_5b = inception(inception_5a, [384, 192, 384, 48, 128, 128], scope='inception_5b')
        
        pool = avg_pool(inception_5b, 7, 1, scope='average_pool')
        
        main_flatten = flatten(pool)
        
        # dropout = tf.nn.dropout(main_flatten, keep_prob=0.4)
        dropout = tf.layers.dropout(main_flatten, rate=dropout, training=is_training)
        
        fc = fully_connected(dropout, 1000, name='fc')
        fc = relu(fc)
        
        main_out = fully_connected(fc, 100, name='output')
        
        return ax1_out, ax2_out, main_out

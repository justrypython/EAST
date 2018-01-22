#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

checkpoint_path = 'tmp/east_icdar2015_resnet_v1_50_rbox/'

def angle_ckpt_test():
    old_checkpoint_file = checkpoint_path + 'model_fapiao_512.ckpt-205036'
    new_checkpoint_file = checkpoint_path + 'model_angle_512.ckpt'
    
    var_names = {'feature_fusion/Conv_9/weights':'feature_fusion/Conv_10/weights',
                 'feature_fusion/Conv_9/biases':'feature_fusion/Conv_10/biases'}
    new_checkpoing_vars = {}
    reader = tf.train.NewCheckpointReader(old_checkpoint_file)
    for old_name in reader.get_variable_to_shape_map():
        new_checkpoing_vars[old_name] = tf.Variable(reader.get_tensor(old_name))
        if 'feature_fusion/Conv_9/weights' in old_name or 'feature_fusion/Conv_9/biases' in old_name:
            new_checkpoing_vars[old_name.replace('Conv_9', 'Conv_10')] = tf.Variable(reader.get_tensor(old_name))
    
    saver = tf.train.Saver(new_checkpoing_vars)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_checkpoint_file)
    
    print('end')

def angle_test():
    geo_map = tf.placeholder(tf.float32, shape=(None, None, None, 4), name='geo_map')
    angle_map_0 = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='angle_map_0')
    angle_map_1 = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='angle_map_1')
    geo_a, geo_b = geo_map[:, :, :, ::2], geo_map[:, :, :, 1::2]
    geo_a_sum = tf.reduce_sum(geo_a, axis=-1)
    geo_b_sum = tf.reduce_sum(geo_b, axis=-1)
    geo_greater = tf.greater(geo_a_sum, geo_b_sum)
    geo_greater = tf.expand_dims(geo_greater, -1)
    angle_map = tf.where(geo_greater, angle_map_0, angle_map_1)
    print('end')
    

if __name__ == '__main__':
    #angle_test()
    angle_ckpt_test()
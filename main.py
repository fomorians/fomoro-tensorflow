from __future__ import print_function
from __future__ import division

from tqdm import tqdm

import numpy as np
import tensorflow as tf

# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from dataset import read_data_sets

def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
    return W, b

def conv2d(x, kernel, output_depth):
    input_depth = x.get_shape().as_list()[-1]
    W, b = weight_bias(kernel + [input_depth, output_depth])
    return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dense(x, output_size, activation):
    input_size = x.get_shape().as_list()[-1]
    W, b = weight_bias([input_size, output_size])
    return activation(tf.matmul(x, W) + b)

with tf.Session() as sess:
    NUM_EPOCHS = 5
    BATCH_SIZE = 64

    # dataset = read_data_sets('mnist', one_hot=True)
    dataset = read_data_sets('clock_dataset.pkl', 'https://s3.amazonaws.com/fomoro-public-datasets/clock_dataset.pkl', dataset_hash='2c53ed06fffb4655426979af44d9e957')

    input_size = dataset.train.images.shape[1]
    output_size = dataset.train.labels.shape[1]
    width, height = 24, 24

    x = tf.placeholder('float', shape=[None, input_size], name='x')
    y_ = tf.placeholder('float', shape=[None, output_size], name='y_')
    keep_prob = tf.placeholder('float', name='keep_prob')

    x_image = tf.reshape(x, [-1, width, height, 1])

    h_conv1 = conv2d(x_image, [3, 3], 32)
    h_pool1 = max_pool(h_conv1)

    h_conv2 = conv2d(h_pool1, [3, 3], 64)
    h_pool2 = max_pool(h_conv2)

    h_pool2_shape = h_pool2.get_shape().as_list()[1:]
    h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(h_pool2_shape)])

    h_fc1 = dense(h_pool2_flat, 128, tf.nn.relu)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y = dense(h_fc1_drop, output_size, tf.nn.softmax)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-7), reduction_indices=[1])
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    num_batches = dataset.train.num_examples // BATCH_SIZE
    for epoch in range(NUM_EPOCHS):
        for batch_index in tqdm(range(num_batches), total=num_batches):
            xs, ys = dataset.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={
                x: xs,
                y_: ys,
                keep_prob: 0.5
            })

        loss_valid, accuracy_valid = sess.run([loss, accuracy], feed_dict={
            x: dataset.validation.images,
            y_: dataset.validation.labels,
            keep_prob: 1.0
        })
        print('[valid] loss: {}, accuracy: {} ({}/{})'.format(loss_valid, accuracy_valid, epoch + 1, NUM_EPOCHS))

    loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict={
        x: dataset.test.images,
        y_: dataset.test.labels,
        keep_prob: 1.0
    })
    print('[test] loss: {}, accuracy: {}'.format(loss_test, accuracy_test))

from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

mnist = input_data.read_data_sets('mnist', one_hot=True)

batch_size = 50

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Graph().as_default(), tf.Session() as sess:
    x = tf.placeholder('float', shape=[None, 784], name='x')
    y_ = tf.placeholder('float', shape=[None, 10], name='y_')

    # reshape input
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully-connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder('float', name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_softmax = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # define training and accuracy operations
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_softmax))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
    accuracy_summary = tf.scalar_summary('accuracy', accuracy)

    merged_summaries = tf.merge_all_summaries()

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver()

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, 'convnet.pb', as_text=False)

    # restore variables
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    if not FLAGS.skip_training:
        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)

        num_steps = 20000
        checkpoint_interval = 100

        step = 0
        for i in range(num_steps):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            if step % checkpoint_interval == 0:
                validation_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={
                    x: mnist.validation.images,
                    y_: mnist.validation.labels,
                    keep_prob: 1.0
                })
                summary_writer.add_summary(summary, step)
                saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)
                print('step %d, training accuracy %g' % (step, validation_accuracy))

            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob: 0.5
            })

            step += 1

        summary_writer.close()

    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    })
    print('test accuracy %g' % test_accuracy)

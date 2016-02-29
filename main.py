from __future__ import print_function
print('MAIN...')

import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', True, 'If true, train the model locally.')
flags.DEFINE_boolean('test', False, 'If true, test the model locally.')
flags.DEFINE_boolean('save', False, 'If true, save the model graph.')
flags.DEFINE_boolean('kaggle', False, 'If true, write predictions against the kaggle test set.')

if FLAGS.train or FLAGS.test:
    print('LOADING MNIST...')
    mnist = input_data.read_data_sets('mnist', one_hot=True)

if FLAGS.kaggle:
    # TODO: wget https://www.kaggle.com/c/digit-recognizer/download/train.csv
    # TODO: wget https://www.kaggle.com/c/digit-recognizer/download/test.csv
    kaggle_test = pd.read_csv('test.csv', sep=',')

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

    if FLAGS.test:
        # create a saver instance to restore from the checkpoint
        saver = tf.train.Saver()

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    if FLAGS.save:
        # save the graph definition as a protobuf file
        tf.train.write_graph(sess.graph_def, 'models/', 'convnet.pb', as_text=False)

    if FLAGS.test:
        # restore variables
        saver.restore(sess, 'checkpoints/latest.ckpt')

    if FLAGS.train:
        summary_writer = tf.train.SummaryWriter('logs/', sess.graph_def)

        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(50)

            if i % 100 == 0:
                validation_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={
                    x: mnist.validation.images,
                    y_: mnist.validation.labels,
                    keep_prob: 1.0
                })
                summary_writer.add_summary(summary, i)
                print('step %d, training accuracy %g' % (i, validation_accuracy))

            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob: 0.5
            })

        summary_writer.close()

    if FLAGS.test:
        test_accuracy = sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.0
        })
        print('test accuracy %g' % test_accuracy)

    if FLAGS.kaggle:
        import pandas as pd

        # gather predictions on kaggle test set
        predictions = sess.run(tf.argmax(y_softmax, 1), feed_dict={
            x: kaggle_test.values,
            keep_prob: 1.0
        })

        # save predictions as csv
        index = np.arange(1, kaggle_test.shape[0] + 1)
        results = pd.DataFrame(predictions, index=index)
        results.to_csv('results.csv', header=['Label'], index=True, index_label='ImageId')

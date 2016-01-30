import tensorflow as tf

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Graph().as_default(), tf.Session() as sess:
    x = tf.placeholder("float", shape=[None, 784], name='x')
    y_ = tf.placeholder("float", shape=[None, 10], name='y_')

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
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_softmax = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # define training and accuracy operations
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_softmax))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_step')
    correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, 'models/', 'convnet.pb', as_text=False)

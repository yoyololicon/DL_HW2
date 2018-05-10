import tensorflow as tf


def Lenet_basic(features, num_class, regularizer):
    # input 64 x 64
    with tf.variable_scope('basic_model'):
        conv1 = tf.layers.conv2d(inputs=features, filters=16, kernel_size=5, padding='valid',
                                 activation=tf.nn.relu, name='conv1', kernel_regularizer=regularizer)  # 60
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='valid', name='pool1')  # 30

        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5, padding='valid', activation=tf.nn.relu,
                                 name='conv2', kernel_regularizer=regularizer)  # 26
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='valid', name='pool2')  # 13

        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv3', kernel_regularizer=regularizer)  # 11
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='valid', name='pool3')  # 5

        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv4', kernel_regularizer=regularizer)  # 3
        # pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding='valid', name='pool4')  # 3

        flatten = tf.layers.flatten(conv4, name='flatten')  # 9 * 128
        dropout = tf.layers.dropout(flatten, 0.2, name='dropout')
        dense = tf.layers.dense(dropout, 1024, activation=tf.nn.relu, kernel_regularizer=regularizer, name='dense')
        logits = tf.layers.dense(inputs=dense, units=num_class, name='outputs', kernel_regularizer=regularizer)
        return logits


def Lenet_basic2(features, num_class, regularizer):
    # input 64^2
    with tf.variable_scope('basic_model2'):
        conv1 = tf.layers.conv2d(inputs=features, filters=16, kernel_size=3, padding='valid',
                                 activation=tf.nn.relu, name='conv1', kernel_regularizer=regularizer)  # 62
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding='valid', name='pool1')  # 31

        conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv2', kernel_regularizer=regularizer)  # 29
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding='valid', name='pool2')  # 14

        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv3', kernel_regularizer=regularizer)  # 12
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='valid', name='pool3')  # 6

        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv4', kernel_regularizer=regularizer)  # 4
        # pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding='valid', name='pool4')

        flatten = tf.layers.flatten(conv4, name='flatten')  # 16 * 128
        dropout = tf.layers.dropout(flatten, 0.2, name='dropout')
        dense = tf.layers.dense(dropout, 1024, activation=tf.nn.relu, kernel_regularizer=regularizer, name='dense')
        logits = tf.layers.dense(inputs=dense, units=num_class, name='outputs', kernel_regularizer=regularizer)
        return logits


def Lenet_basic3(features, num_class, regularizer):
    # input 64^2
    with tf.variable_scope('basic_model3'):
        conv1 = tf.layers.conv2d(inputs=features, filters=16, kernel_size=5, strides=2, padding='valid',
                                 activation=tf.nn.relu, name='conv1', kernel_regularizer=regularizer)  # 60

        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=5, strides=2, padding='valid',
                                 activation=tf.nn.relu,
                                 name='conv2', kernel_regularizer=regularizer)  # 26

        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=2, padding='valid',
                                 activation=tf.nn.relu,
                                 name='conv3', kernel_regularizer=regularizer)  # 11

        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=3, padding='valid', activation=tf.nn.relu,
                                 name='conv4', kernel_regularizer=regularizer)  # 3
        flatten = tf.layers.flatten(conv4, name='flatten')  # 9 * 128
        dropout = tf.layers.dropout(flatten, 0.2, name='dropout')
        dense = tf.layers.dense(dropout, 1024, activation=tf.nn.relu, kernel_regularizer=regularizer, name='dense')
        logits = tf.layers.dense(inputs=dense, units=num_class, name='outputs', kernel_regularizer=regularizer)

        return logits


def resnet18_plain_simple(features, num_class, regularizer):
    # input should be 64^2
    with tf.variable_scope('resnet18_plain_simple'):
        # Tensor for input layer protocol
        conv1 = tf.layers.conv2d(inputs=features, filters=64, kernel_size=[5, 5], strides=2, padding='valid',
                                 activation=tf.nn.relu, kernel_regularizer=regularizer)
        conv2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding='valid')

        for _ in range(4):
            conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], padding='same',
                                     activation=tf.nn.relu, kernel_regularizer=regularizer)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], strides=2, padding='valid',
                                 activation=tf.nn.relu, kernel_regularizer=regularizer)  # 8 x 8

        for _ in range(3):
            conv3 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], padding='same',
                                     activation=tf.nn.relu, kernel_regularizer=regularizer)

        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], strides=2, padding='valid',
                                 activation=tf.nn.relu, kernel_regularizer=regularizer)  # 4 x 4
        for _ in range(3):
            conv4 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], padding='same',
                                     activation=tf.nn.relu, kernel_regularizer=regularizer)

        # average pooling
        average = tf.reduce_mean(conv4, [1, 2], keep_dims=True)
        average = tf.reshape(average, [-1, 256])

        logits = tf.layers.dense(inputs=average, units=num_class, kernel_regularizer=regularizer)
        return logits


def rnn(features, num_hidden, num_class, init_states):
    with tf.variable_scope('myrnn'):
        cells = [tf.contrib.rnn.BasicRNNCell(n) for n in num_hidden]
        stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        rnn_outputs, final_states = tf.nn.dynamic_rnn(stacked_rnn_cell, features, initial_state=init_states)
        logits = tf.layers.dense(inputs=rnn_outputs, units=num_class)
        return logits, final_states
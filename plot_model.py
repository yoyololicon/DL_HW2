import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('l2_regular/model-2-3900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('l2_regular/'))

    c1_weights = sess.graph.get_tensor_by_name('basic_model2/conv1/kernel:0').eval()
    c2_weights = sess.graph.get_tensor_by_name('basic_model2/conv2/kernel:0').eval()
    c3_weights = sess.graph.get_tensor_by_name('basic_model2/conv3/kernel:0').eval()
    c4_weights = sess.graph.get_tensor_by_name('basic_model2/conv4/kernel:0').eval()

    c1_bias = sess.graph.get_tensor_by_name('basic_model2/conv1/bias:0').eval()
    c2_bias = sess.graph.get_tensor_by_name('basic_model2/conv2/bias:0').eval()
    c3_bias = sess.graph.get_tensor_by_name('basic_model2/conv3/bias:0').eval()
    c4_bias = sess.graph.get_tensor_by_name('basic_model2/conv4/bias:0').eval()

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 9)

    ax[0, 0].hist(c1_weights.flatten(), bins=50)
    ax[0, 0].set_title("Histogram of conv1 kernel weights")
    ax[0, 0].set_ylabel("Number")
    #ax[0, 0].set_xlim(-1.5, 1.5)

    ax[0, 1].hist(c2_weights.flatten(), bins=50)
    ax[0, 1].set_title("Histogram of conv2 kernel weights")
    #ax[0, 1].set_xlim(-1.5, 1.5)

    ax[1, 0].hist(c3_weights.flatten(), bins=50)
    ax[1, 0].set_title("Histogram of conv3 kernel weights")
    ax[1, 0].set_xlabel("Value")
    ax[1, 0].set_ylabel("Number")
    #ax[1, 0].set_xlim(-1.5, 1.5)

    ax[1, 1].hist(c4_weights.flatten(), bins=50)
    ax[1, 1].set_title("Histogram of conv4 kernel weights")
    ax[1, 1].set_xlabel("Value")
    #ax[1, 1].set_xlim(-1.5, 1.5)

    plt.show()

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 9)

    ax[0, 0].hist(c1_bias)
    ax[0, 0].set_title("Histogram of conv1 bias")
    ax[0, 0].set_ylabel("Number")
    #ax[0, 0].set_xlim(-1.5, 1.5)

    ax[0, 1].hist(c2_bias)
    ax[0, 1].set_title("Histogram of conv2 bias")
    #ax[0, 1].set_xlim(-1.5, 1.5)

    ax[1, 0].hist(c3_bias)
    ax[1, 0].set_title("Histogram of conv3 bias")
    ax[1, 0].set_xlabel("Value")
    ax[1, 0].set_ylabel("Number")
    #ax[1, 0].set_xlim(-1.5, 1.5)

    ax[1, 1].hist(c4_bias)
    ax[1, 1].set_title("Histogram of conv4 bias")
    ax[1, 1].set_xlabel("Value")
    #ax[1, 1].set_xlim(-1.5, 1.5)

    plt.show()

    out_weights = sess.graph.get_tensor_by_name('basic_model2/outputs/kernel:0').eval()
    out_bias = sess.graph.get_tensor_by_name('basic_model2/outputs/bias:0').eval()

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    ax[0].hist(out_weights.flatten(), bins=50)
    ax[0].set_title("Histogram of output layer weights")
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Number")

    ax[1].hist(out_bias)
    ax[1].set_title("Histogram of output layer bias")
    ax[1].set_xlabel("Value")
    plt.show()
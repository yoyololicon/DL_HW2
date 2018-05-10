import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import itertools
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from datetime import datetime
from random import shuffle
from utils import target_name, plot_hidden_feature

data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/Food-11/evaluation'
scale_height = scale_width = 64
num_class = 11
batch_size = 256


def read_img(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [scale_height, scale_width])
    image_std = tf.image.per_image_standardization(image_resized)
    return image_std, tf.one_hot(label, num_class)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


eva_files = os.listdir(data_dir)
eva_num = len(eva_files)
eva_labels = tf.constant([int(f.split("_")[0]) for f in eva_files])
eva_files = tf.constant([os.path.join(data_dir, f) for f in eva_files])
eva_data = tf.data.Dataset.from_tensor_slices((eva_files, eva_labels))
eva_data = eva_data.map(read_img)
features, labels = eva_data.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('fast_model2/model-2-3900.meta')
    saver.restore(sess, tf.train.latest_checkpoint('fast_model2/'))

    predict_op = sess.graph.get_tensor_by_name('predict_op:0')
    X = sess.graph.get_tensor_by_name('input_data:0')
    Y = sess.graph.get_tensor_by_name('output_target:0')


    eva_x = []
    eva_y = []
    print("Reading", eva_num, "of evaluation files ...")
    a = datetime.now().replace(microsecond=0)
    while True:
        try:
            imgs, clss = sess.run([features, labels])
            eva_x.append(imgs[None, :])
            eva_y.append(clss)
        except tf.errors.OutOfRangeError:
            break
    print("Complete reading validation files.")
    print("Time cost:", datetime.now().replace(microsecond=0) - a)
    eva_x = np.row_stack(eva_x)
    eva_y = np.row_stack(eva_y)

    y_pred = []
    # calculate misclassification
    for i in range(eva_num // batch_size + 1):
        pos = i * batch_size
        nums = min(eva_num, pos + batch_size) - pos
        y = sess.run(predict_op, feed_dict={X: eva_x[pos:pos + nums], Y: eva_y[pos:pos + nums]})
        y_pred.append(y)

    y_pred = np.concatenate(y_pred)
    y_true = np.argmax(eva_y, axis=1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    plot_confusion_matrix(cm, list(target_name.values()))
    plt.show()


    conv1 = sess.graph.get_tensor_by_name('basic_model2/conv1/Relu:0')
    conv2 = sess.graph.get_tensor_by_name('basic_model2/conv2/Relu:0')
    conv3 = sess.graph.get_tensor_by_name('basic_model2/conv3/Relu:0')
    conv4 = sess.graph.get_tensor_by_name('basic_model2/conv4/Relu:0')

    file_names = os.listdir(data_dir)
    shuffle(file_names)
    for f in file_names:
        label = np.zeros(num_class)
        label[int(f.split("_")[0])] = 1
        image_string = tf.read_file(os.path.join(data_dir, f))
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_resized = tf.image.resize_images(image_float, [scale_height, scale_width])
        image_std = tf.image.per_image_standardization(image_resized)

        y, hidden1, hidden2, hidden3, hidden4 = sess.run([predict_op, conv1, conv2, conv3, conv4],
                                                         feed_dict={X: image_std.eval()[None, :], Y: label[None, :]})
        y = y[0]
        t = np.argmax(label)
        if (y == 1 or y == 2) and (t == 1 or t == 2):
            plt.imshow(np.asarray(image_decoded.eval()))
            plt.title('Predict: ' + target_name[y] + ', Label: ' + target_name[t])
            plt.show()

            plot_hidden_feature(hidden1, 4, title='conv1 feature maps')
            plot_hidden_feature(hidden2, 6, title='conv2 feature maps')
            # plot_hidden_feature(hidden3, 8, title='conv3 feature maps')
            # plot_hidden_feature(hidden4, 12, title='conv4 feature maps')

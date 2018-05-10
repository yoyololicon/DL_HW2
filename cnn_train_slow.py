import matplotlib
matplotlib.use('Agg')

from model import Lenet_basic2, Lenet_basic, Lenet_basic3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

data_dir = '/home/0316223/Datasets/Food-11'
train_dir = os.path.join(data_dir, 'training')
val_dir = os.path.join(data_dir, 'validation')
eva_dir = os.path.join(data_dir, 'evaluation')

scale_height = scale_width = 64
# parameters for training
batch_size = 256
epochs = 80
num_class = 11
init_learning_rate = 0.01

weight_decay = 0.0

# basic model Evaluation accuracy: 0.45593068424822647


def read_img(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [scale_height, scale_width])
    image_std = tf.image.per_image_standardization(image_resized)
    return image_std, tf.one_hot(label, num_class)


def distorted_img(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [72, 72])

    image_crop = tf.random_crop(image_resized, [scale_height, scale_width, 3])
    image_flip = tf.image.random_flip_left_right(image_crop)

    image_std = tf.image.per_image_standardization(image_flip)
    return image_std, tf.one_hot(label, num_class)


train_files = [f for f in os.listdir(train_dir)]
train_num = len(train_files)

train_labels = tf.constant([int(f.split("_")[0]) for f in train_files])
train_files = tf.constant([os.path.join(train_dir, f) for f in train_files])

train_data = tf.data.Dataset.from_tensor_slices((train_files, train_labels))

clean_data = train_data.map(read_img)
distorted_data = train_data.map(distorted_img)
train_data = clean_data.concatenate(distorted_data)
# train_data = clean_data
train_data = train_data.shuffle(1000)
train_data = train_data.repeat()
train_data = train_data.batch(batch_size)

steps_per_epoch = train_num * 2 // batch_size

val_files = [f for f in os.listdir(val_dir)]
val_num = len(val_files)
val_labels = tf.constant([int(f.split("_")[0]) for f in val_files])
val_files = tf.constant([os.path.join(val_dir, f) for f in val_files])
val_data = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

val_data = val_data.map(read_img)
val_data = val_data.repeat()
val_data = val_data.batch(batch_size)

eva_files = [f for f in os.listdir(eva_dir)]
eva_num = len(eva_files)
eva_labels = tf.constant([int(f.split("_")[0]) for f in eva_files])
eva_files = tf.constant([os.path.join(eva_dir, f) for f in eva_files])
eva_data = tf.data.Dataset.from_tensor_slices((eva_files, eva_labels))

eva_data = eva_data.map(read_img)
eva_data = eva_data.padded_batch(batch_size, padded_shapes=([scale_height, scale_width, 3], [num_class]))

# this one is for future use when we need extra input
X = tf.placeholder(dtype=tf.float32, shape=[None, scale_height, scale_width, 3], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_class], name='Y')
input_data = tf.data.Dataset.from_tensor_slices((X, Y))
input_data = input_data.batch(1)

# create common iterator
iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
features, labels = iter.get_next()

# create initialisation operation
train_init_op = iter.make_initializer(train_data)
val_init_op = iter.make_initializer(val_data)
eva_init_op = iter.make_initializer(eva_data)
input_init_op = iter.make_initializer(input_data, name='input_init_op')

def main(_):
    with tf.Session() as sess:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        # define your own fully connected DNN
        outputs = Lenet_basic3(features, num_class, regularizer)

        # tensor for prediction the class
        predicts = tf.argmax(outputs, -1, name='predict_op')

        # Add training ops into graph.
        with tf.variable_scope('train'):
            # tensor for calculate loss by softmax cross-entroppy
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs), name='loss_op')

            loss += tf.losses.get_regularization_loss()
            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            #boundaries = [2000, 4000]
            #values = [0.1, 0.05, 0.02]
            #learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 2000, 0.5, staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

            arg_labels = tf.argmax(labels, -1)
            acc = tf.reduce_mean(tf.cast(tf.equal(predicts, arg_labels), tf.float32), name='acc_op')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # Assign the required tensors to do the operation

        saver = tf.train.Saver()

        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        acc_op = sess.graph.get_tensor_by_name('train/acc_op:0')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')

        # Start training
        print('Start training on', train_num, 'of samples ...')
        print('Validate on', val_num, 'of samples.')
        start = datetime.now().replace(microsecond=0)
        loss_history = []
        acc_history = []
        for i in range(epochs):
            # switch to training set
            sess.run(train_init_op)
            total_loss = 0
            total_acc = 0
            for _ in range(steps_per_epoch):
                _, loss_value, acc_value = sess.run([train_op, loss_tensor, acc_op])
                total_loss += loss_value
                total_acc += acc_value

                if global_step_tensor.eval() % 1000 == 0:
                    saver.save(sess, './model-3', global_step=global_step_tensor)

            total_loss /= steps_per_epoch
            total_acc /= steps_per_epoch
            print("Iter: {}, Global step: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(i, global_step_tensor.eval(),
                                                                                     total_loss, total_acc))
            loss_history.append(total_loss)
            # switch to validation set
            sess.run(val_init_op)
            val_acc = 0
            for _ in range(val_num // batch_size):
                acc_value = sess.run(acc_op)
                val_acc += acc_value
            val_acc /= val_num // batch_size
            print("Validation accuracy:", val_acc)
            acc_history.append([total_acc, val_acc])
        
        end = datetime.now().replace(microsecond=0)

        print("Time cost:", end - start)
        saver.save(sess, './model-3', global_step=global_step_tensor)
        
        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("Cross-entroppy loss")
        plt.title("Training curve")
        plt.savefig("model1_curve.png", dpi=100)

        plt.gcf().clear()

        acc_history = np.array(acc_history).T
        plt.plot(acc_history[0], label='training accuracy')
        plt.plot(acc_history[1], label='validation accuracy')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy rate")
        plt.title("Training accuracy")
        plt.legend()
        plt.savefig("Training accuracy", dpi=100)

        print('Evaluate on', eva_num, 'of samples.')
        sess.run(eva_init_op)
        eva_acc = 0
        steps = eva_num // batch_size
        rem_batch = eva_num % batch_size
        for _ in range(steps):
            acc_value = sess.run(acc_op)
            eva_acc += acc_value
        eva_acc = (eva_acc * batch_size + sess.run(acc_op) * rem_batch) / eva_num
        print("Evaluation accuracy:", eva_acc)


if __name__ == '__main__':
    tf.app.run()

import matplotlib
matplotlib.use('Agg')

from model import Lenet_basic2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.utils import shuffle

data_dir = '/home/0316223/Datasets/Food-11'
train_dir = os.path.join(data_dir, 'training')
val_dir = os.path.join(data_dir, 'validation')
eva_dir = os.path.join(data_dir, 'evaluation')
model_name = 'model-2'
scale_height = scale_width = 64
# parameters for training
batch_size = 256
epochs = 50
num_class = 11
init_learning_rate = 0.01
weight_decay = 0.01

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

print("Image preprocessing...")
#typical dataset usage
train_files = [f for f in os.listdir(train_dir)]
train_num = len(train_files)
train_labels = tf.constant([int(f.split("_")[0]) for f in train_files])
train_files = tf.constant([os.path.join(train_dir, f) for f in train_files])
train_data = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
clean_data = train_data.map(read_img)
distorted_data = train_data.map(distorted_img)
train_data = clean_data.concatenate(distorted_data)
train_data = train_data.shuffle(1000)
train_num *= 2
steps_per_epoch = train_num // batch_size + 1

val_files = [f for f in os.listdir(val_dir)]
val_num = len(val_files)
val_labels = tf.constant([int(f.split("_")[0]) for f in val_files])
val_files = tf.constant([os.path.join(val_dir, f) for f in val_files])
val_data = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_data = val_data.map(read_img)

eva_files = [f for f in os.listdir(eva_dir)]
eva_num = len(eva_files)
eva_labels = tf.constant([int(f.split("_")[0]) for f in eva_files])
eva_files = tf.constant([os.path.join(eva_dir, f) for f in eva_files])
eva_data = tf.data.Dataset.from_tensor_slices((eva_files, eva_labels))
eva_data = eva_data.map(read_img)

iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
features, labels = iter.get_next()

train_init_op = iter.make_initializer(train_data)
val_init_op = iter.make_initializer(val_data)
eva_init_op = iter.make_initializer(eva_data)

X = tf.placeholder(dtype=tf.float32, shape=[None, scale_height, scale_width, 3], name='input_data')
Y = tf.placeholder(dtype=tf.float32, shape=[None, num_class], name='output_target')

def main(_):
    with tf.Session() as sess:
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        eva_x = []
        eva_y = []
        
        print("Start reading", train_num, "of training files ...")
        a = datetime.now().replace(microsecond=0)
        sess.run(train_init_op)
        while True:
            try:
                imgs, clss = sess.run([features, labels])
                train_x.append(imgs[None, :])
                train_y.append(clss)
            except tf.errors.OutOfRangeError:
                break
        b = datetime.now().replace(microsecond=0)
        print("Complete reading training files.")
        print("Time cost:", b-a)
        a = b

        print("Start reading", val_num, "of validation files ...")
        sess.run(val_init_op)
        while True:
            try:
                imgs, clss = sess.run([features, labels])
                val_x.append(imgs[None, :])
                val_y.append(clss)
            except tf.errors.OutOfRangeError:
                break
        b = datetime.now().replace(microsecond=0)
        print("Complete reading validation files.")
        print("Time cost:", b-a)
        a = b

        print("Start reading", eva_num, "of evaluation files ...")
        sess.run(eva_init_op)
        while True:
            try:
                imgs, clss = sess.run([features, labels])
                eva_x.append(imgs[None, :])
                eva_y.append(clss)
            except tf.errors.OutOfRangeError:
                break
        b = datetime.now().replace(microsecond=0)
        print("Complete reading validation files.")
        print("Time cost:", b-a)

        train_x = np.row_stack(train_x)
        train_y = np.row_stack(train_y)
        val_x = np.row_stack(val_x)
        val_y = np.row_stack(val_y)
        eva_x = np.row_stack(eva_x)
        eva_y = np.row_stack(eva_y)
        
        #construct model
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        # define your own fully connected DNN
        outputs = Lenet_basic2(X, num_class, regularizer)

        # tensor for prediction the class
        predicts = tf.argmax(outputs, -1, name='predict_op')

        # Add training ops into graph.
        with tf.variable_scope('train'):
            # tensor for calculate loss by softmax cross-entroppy
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=outputs), name='loss_op')

            loss += tf.losses.get_regularization_loss()
            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            #learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 2000, 0.5, staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate=init_learning_rate, momentum=0.9)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

            arg_labels = tf.argmax(Y, -1)
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
        print('Start training ...')
        a = datetime.now().replace(microsecond=0)
        loss_history = []
        acc_history = []
        for i in range(epochs):
            total_loss = 0
            total_acc = 0
            temp_x, temp_y = shuffle(train_x, train_y)
            for j in range(steps_per_epoch):
                pos = j* batch_size
                nums = min(train_num, pos + batch_size) - pos
                _, loss_value, acc_value = sess.run([train_op, loss_tensor, acc_op], 
                                                    feed_dict={X: temp_x[pos:pos+nums], Y: temp_y[pos:pos+nums]})
                total_loss += loss_value*nums
                total_acc += acc_value*nums

                if global_step_tensor.eval() % 1000 == 0:
                    saver.save(sess, './'+model_name, global_step=global_step_tensor)

            total_loss /= train_num
            total_acc /= train_num

            val_acc = 0
            val_loss = 0
            for j in range(val_num // batch_size+1):
                pos = j* batch_size
                nums = min(val_num, pos + batch_size) - pos
                loss_value, acc_value = sess.run([loss_tensor, acc_op], feed_dict={X: val_x[pos:pos+nums], Y: val_y[pos:pos+nums]})
                val_acc += acc_value*nums
                val_loss += loss_value*nums
            val_acc /= val_num
            val_loss /= val_num
            print("Iter: {}, Global step: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
                    .format(i, global_step_tensor.eval(), total_loss, total_acc, val_loss, val_acc))
            acc_history.append([total_acc, val_acc])
            loss_history.append([total_loss, val_loss])
        
        b = datetime.now().replace(microsecond=0)

        print("Time cost:", b - a)
        saver.save(sess, './'+model_name, global_step=global_step_tensor)
        
        loss_history = np.array(loss_history).T
        plt.plot(loss_history[0], label='training loss')
        plt.plot(loss_history[1], label='validation loss')
        plt.xlabel("epochs")
        plt.ylabel("Cross-entroppy loss")
        plt.title("Training curve")
        plt.legend()
        plt.savefig("Training curve.png", dpi=100)

        plt.gcf().clear()

        acc_history = np.array(acc_history).T
        plt.plot(acc_history[0], label='training accuracy')
        plt.plot(acc_history[1], label='validation accuracy')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy rate")
        plt.title("Training accuracy")
        plt.legend()
        plt.savefig("Training accuracy", dpi=100)

        eva_acc = 0
        for i in range(eva_num//batch_size + 1):
            pos = i* batch_size
            nums = min(eva_num, pos + batch_size) - pos
            acc_value = sess.run(acc_op, feed_dict={X: eva_x[pos:pos+nums], Y: eva_y[pos:pos+nums]})
            eva_acc += acc_value*nums
        eva_acc /= eva_num
        print("Evaluation accuracy:", eva_acc)

if __name__ == '__main__':
    tf.app.run()

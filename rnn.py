from data_utils import get_batches, train_encode, vocab, vocab_to_int, int_to_vocab
import numpy as np
import tensorflow as tf
from model import rnn
import matplotlib.pyplot as plt

seq = np.array(['N', 'C', 'T', 'U', ' ', 'i', 's', ' ', 'g', 'o', 'o', 'd'])

valid_URL = 'Dataset/shakespeare_valid.txt'
with open(valid_URL, 'r') as f:
    text = f.read()
valid_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

batch_size = 32
num_steps = 100
num_hidden = [128, 128]
num_class = len(vocab)
learning_rate = 0.1
epochs = 20

X = tf.placeholder(tf.int32, [None, num_steps], name='input_X')
Y = tf.placeholder(tf.int32, [None, num_steps], name='labels_Y')
rnn_inputs = tf.one_hot(X, num_class)
labels = tf.one_hot(Y, num_class)

init_states = tuple(tf.placeholder(tf.float32, [None, n]) for n in num_hidden)


def main(_):
    with tf.Session() as sess:
        outputs, final_states = rnn(rnn_inputs, num_hidden, num_class, init_states)
        predicts = tf.argmax(outputs, -1, name='predict_op')
        with tf.variable_scope('train'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs),
                                  name='loss_op')

            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

            arg_labels = tf.argmax(labels, -1)
            acc = tf.reduce_mean(tf.cast(tf.equal(predicts, arg_labels), tf.float32), name='acc_op')

        sess.run(tf.global_variables_initializer())
        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        acc_op = sess.graph.get_tensor_by_name('train/acc_op:0')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')

        print('Start training ...')
        loss_history = []
        acc_history = []
        for i in range(epochs):
            total_loss = 0
            total_acc = 0
            count = 0
            current_states = tuple(np.zeros((batch_size, n)) for n in num_hidden)
            for x, y in get_batches(train_encode, batch_size, num_steps):
                _, loss_value, acc_value, current_states = sess.run([train_op, loss_tensor, acc_op, final_states],
                                                                    feed_dict={X: x, Y: y, init_states: current_states})
                total_loss += loss_value
                total_acc += acc_value
                count += 1
            total_loss /= count
            total_acc /= count

            valid_acc = 0
            count = 0
            current_states = tuple(np.zeros((batch_size, n)) for n in num_hidden)
            for x, y in get_batches(valid_encode, batch_size, num_steps):
                acc_value, current_states = sess.run([acc_op, final_states],
                                                     feed_dict={X: x, Y: y, init_states: current_states})
                valid_acc += acc_value
                count += 1
            valid_acc /= count
            print("Average", total_loss, total_acc, valid_acc)
            loss_history.append(total_loss)
            acc_history.append([total_acc, valid_acc])

        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("BPC")
        plt.title("Training curve")
        plt.show()

        acc_history = np.array(acc_history).T
        err_history = 1 - acc_history
        plt.plot(err_history[0], label='training error')
        plt.plot(err_history[1], label='validation error')
        plt.xlabel("epochs")
        plt.ylabel("Error rate")
        plt.title("Training error")
        plt.legend()
        plt.show()


        # predict 100 words
        for x, y in get_batches(train_encode, 1, num_steps):
            seed_encode = x[0]
            break
        seed = [int_to_vocab[c] for c in seed_encode]
        seed = ''.join(seed)
        current_states = tuple(np.zeros((1, n)) for n in num_hidden)
        for i in range(10):
            if i > 0:
                seed_encode = np.append(last, np.zeros(num_steps - 1))
                for j in range(num_steps-1):
                    predicted = sess.run(predicts, feed_dict={X: seed_encode[None, :], init_states: current_states})
                    seed_encode[j+1] = predicted[0, j]
                    seed += int_to_vocab[predicted[0, j]]
            print(seed_encode)
            predicted, current_states = sess.run([predicts, final_states],
                                                 feed_dict={X: seed_encode[None, :], init_states: current_states})
            last = predicted[0, -1]
            seed += int_to_vocab[last]
        print(seed)


if __name__ == '__main__':
    tf.app.run()

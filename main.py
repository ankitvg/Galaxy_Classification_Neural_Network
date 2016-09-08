# Local libararies
from datahelper import DataHelper
from network import CandleNet
from math import sqrt
import time

# Imports
import tensorflow as tf


# params
epochs = 10
batch_size = 50
test_size = 3878
display_step = 100
test_split = 19000
n_classes = 5
learning_rate = 0.0001  # get numbers from paper?
momentum = tf.constant(0.9)
target_acc = 0.13
model_dir = './models/'
train_progress = './report/train_progress.csv'
test_progress = './report/test_progress.csv'

train = True
checkpoint_dir = './models/'

# input, label placeholders
x = tf.placeholder(tf.float32, [batch_size, 45, 45, 3])
y = tf.placeholder(tf.float32, [None, n_classes])

#tf.set_random_seed(13366)

# create network
net = CandleNet.get_network(x)

# loss and optimizer
# Use squared error, becuase our output doesn't reduce to probability dist
cost = tf.reduce_mean(tf.squared_difference(net, y))
#cost = tf.squared_difference(net, y)
# https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate, the paper uses RMSE
# http://stackoverflow.com/questions/33846069
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(net, y)))

# variable initializer
init = tf.initialize_all_variables()

# model saver
saver = tf.train.Saver()

# train, test, and save model
with tf.Session() as sess:
    sess.run(init)

    if train:
        # train
        epoch = 1
        while True:  # epoch <= epochs:
            epoch_start = time.time()
            print 'Training Epoch {}...'.format(epoch)
            # get data, test_idx = 19000 is ~83% train test split
            dh = DataHelper(batch_size, test_idx=test_split)
            # test data
            test_data, test_labels = dh.get_test_data(test_size)

            step = 1
            while step * batch_size < test_split:
                batch_xs, batch_ys = dh.get_next_batch()

                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                if step % display_step == 0:
                    acc = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

                    print "Iter " + str(step * batch_size) + \
                          ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training RMSE= " + "{:.5f}".format(acc)

                    with open(train_progress, mode='a') as f:
                        f.write('{},{},{},{}\n'.format(epoch,
                                                       (step * batch_size),
                                                       acc,
                                                       loss))

                step += 1

            print 'Saving checkpoint'
            saver.save(sess, model_dir, global_step=epoch)
            print 'Epoch {} finished'.format(epoch)

            print 'Testing...'
            # test
            test_step = 1
            test_rmse = 0.0
            while test_step * batch_size < test_size:
                start = (test_step - 1) * batch_size
                end = test_step * batch_size
                batch_xs = test_data[start:end]
                batch_ys = test_labels[start:end]

                _rmse = sess.run(rmse, feed_dict={x: batch_xs, y: batch_ys})
                _rmse = pow(_rmse, 2) * batch_size
                test_rmse += _rmse

                test_step += 1

            test_rmse = sqrt(test_rmse / float(test_size))
            print 'Test RMSE:{}'.format(test_rmse)

            with open('./report/test_progress.csv', mode='a') as f:
                f.write('{},{}\n'.format(epoch, test_rmse))

            if test_rmse < target_acc:
                break

            print 'Time for epoch {}'.format(time.time() - epoch_start)

            epoch += 1
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        dh = DataHelper(batch_size, test_idx=test_split)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print 'no checkpoint found...'

        batch_xs, _ = dh.get_next_batch()

        predictions = sess.run(net, feed_dict={x: batch_xs})

        print predictions
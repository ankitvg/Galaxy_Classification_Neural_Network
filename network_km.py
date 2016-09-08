
from datahelper import DataHelper
import tensorflow as tf
import sys
import time
# sess = tf.InteractiveSession() # remember to remove

# Network Parameters
IMAGE_SIZE = 45
NUM_LABELS = 5

# Parameters
epochs = 200
learning_rate = 0.0001
batch_size = 200  # batch size has to divide evenly into total example
display_step = 20
test_start = 19000 # index to start testing from
test_size = 1800
dropout_prob = 0.5
train_progress = './report/train_progress.csv'
test_progress = './report/test_progress.csv'
model_dir = './models/'

# Input place holder nodes.
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

tf.set_random_seed(13366)

# Convolution Layer 1

# Weights & biases CL1
W_conv1 = tf.Variable(tf.truncated_normal([6,6,3,32], stddev=0.01))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

print 'ConvPool 1'
print W_conv1.get_shape()
print b_conv1.get_shape()
print conv1.get_shape()
print h_conv1.get_shape()
print pool1.get_shape()


# Convolution Layer 2
# Weights & biases CL2
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.01))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
h_conv2 = tf.nn.relu(conv2 + b_conv2)

pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 2'
print W_conv2.get_shape()
print b_conv2.get_shape()
print conv2.get_shape()
print h_conv2.get_shape()
print pool2.get_shape()



# Convolution Layer 3
# Weights & biases CL3
W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.01))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))

conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
h_conv3 = tf.nn.relu(conv3 + b_conv3)

#pool3 = tf.nn.max_pool(h_conv3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 3'
print W_conv3.get_shape()
print b_conv3.get_shape()
print conv3.get_shape()
print h_conv3.get_shape()
#print pool3.get_shape()


# Convolution Layer 4
# Weights & biases CL4
W_conv4 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.01))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))

conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID')
h_conv4 = tf.nn.relu(conv4 + b_conv4)

pool4 = tf.nn.max_pool(h_conv4, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

print 'ConvPool 4'
print W_conv4.get_shape()
print b_conv4.get_shape()
print conv4.get_shape()
print h_conv4.get_shape()
print pool4.get_shape()

keep_prob = tf.placeholder(tf.float32)


# Fully connected Layer1
W_fc1 = tf.Variable(tf.truncated_normal([2*2*128, 2048], stddev=0.001))
b_fc1 = tf.Variable(tf.constant(0.01, shape=[2048]))

pool4_flat = tf.reshape(pool4, [-1, 2*2*128])
h_fc1 = tf.nn.relu(tf.matmul(pool4_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)


print 'FC 1'
print W_fc1.get_shape()
print b_fc1.get_shape()
print pool4_flat.get_shape()
print h_fc1.get_shape()


# Fully connected Layer2
W_fc2 = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001))
b_fc2 = tf.Variable(tf.constant(0.01, shape=[2048]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2 = tf.nn.dropout(h_fc2, keep_prob)

print 'FC 2'
print W_fc2.get_shape()
print b_fc2.get_shape()
print h_fc2.get_shape()


# Output Layer
W_out = tf.Variable(tf.truncated_normal([2048, NUM_LABELS], stddev=0.001))
b_out = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

out = tf.sigmoid(tf.matmul(h_fc2, W_out) + b_out)

print 'OUT'
print W_out.get_shape()
print b_out.get_shape()
print out.get_shape()


# No changes to old network.py beyond this. Will be updating this soon.


cost = tf.reduce_mean(tf.squared_difference(out, y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(out, y_)))

# Initialize
init = tf.initialize_all_variables()

dh = DataHelper(batch_size, test_idx=test_start)
saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(init)
    print sess.run(W_conv1), sess.run(b_conv1), sess.run(W_conv2), sess.run(b_conv2)
    test_data, test_labels = dh.get_test_data(test_size)
    epoch = 1
    train_start = time.time()
    while epoch <= epochs:
        epoch_start = time.time()
	print 'Training Epoch {}...'.format(epoch)
        # get data, test_idx = 19000 is ~83% train test split
        dh = DataHelper(batch_size, test_idx=test_start)
        # test data
        step = 1
        # Looks like training iters in the number of images to process
        while step * batch_size < test_start:
            # TODO get data in proper format
            batch_xs, batch_ys = dh.get_next_batch()
            #print batch_xs.shape, batch_ys.shape
            #sys.exit(0)
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dropout_prob})

            if step % display_step == 0:
                acc = sess.run(rmse, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                loss = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

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
        epoch_end = time.time()
	print "Time for epoch: ", (epoch_end-epoch_start), " seconds"

	print 'Testing...'
        # test
        test_rmse = sess.run(rmse, feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})

        print 'Average Test RMSE:{}'.format(test_rmse)

        with open('./report/test_progress.csv', mode='a') as f:
            f.write('{},{}\n'.format(epoch, test_rmse))

        epoch += 1

    print "Optimization Finished!"
    train_end = time.time()
    print "Time for full training: ", (train_end - train_start), " seconds"
    print "Testing Error:"
    test_error = sess.run(rmse, feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print test_error


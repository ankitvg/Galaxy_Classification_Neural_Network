# This file contains the network implementation according to the paper
# 5 epochs
# printing average time per epoch

from datahelper import DataHelper
import tensorflow as tf
import sys
import time
from math import sqrt
#import scale_images

# sess = tf.InteractiveSession() # remember to remove
start1 = time.time()
IMAGE_SIZE = 45
NUM_LABELS = 5
learning_rate = 0.004  # get numbers from paper?
batch_size = 50  # batch size has to divide evenly into total example
display_step = 100
test_size=1000
test_split=19000

# Input place holder nodes.
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

# Convolution Layer 1
# Weights & biases CL1
W_conv1 = tf.Variable(tf.truncated_normal([6,6,3,32], stddev=0.01))
b_conv1 = tf.Variable(tf.truncated_normal([32]))
conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
conv1b = tf.nn.bias_add(conv1, b_conv1)
h_conv1 = tf.nn.relu(conv1b)
pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

# Convolution Layer 2
# Weights & biases CL2
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.01))
b_conv2 = tf.Variable(tf.truncated_normal([64]))
conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
conv2b = tf.nn.bias_add(conv2, b_conv2)
h_conv2 = tf.nn.relu(conv2b)
pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

# Convolution Layer 3
# Weights & biases CL3
W_conv3 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.01))
b_conv3 = tf.Variable(tf.truncated_normal([128]))
conv3 = tf.nn.conv2d(pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
conv3b = tf.nn.bias_add(conv3, b_conv3)
h_conv3 = tf.nn.relu(conv3b)

# Convolution Layer 4
# Weights & biases CL4
W_conv4 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.01))
b_conv4 = tf.Variable(tf.truncated_normal([128]))
conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID')
conv4b = tf.nn.bias_add(conv4, b_conv4)
h_conv4 = tf.nn.relu(conv4b)
pool4 = tf.nn.max_pool(h_conv4, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')
pool4_flat = tf.reshape(pool4, [-1, 2*2*128])

# Fully connected Layer1
W_fc1 = tf.Variable(tf.truncated_normal([2*2*128, 2048], stddev=0.001))
b_fc1 = tf.Variable(tf.truncated_normal([2048]))
#h_mat1 = tf.matmul(pool4_flat, W_fc1)
#h_fc1b = tf.nn.bias_add(h_mat1, b_fc1)
#h_fc1 = tf.nn.relu(h_fc1b)
h_fc1 = tf.nn.relu(tf.add(tf.matmul(pool4_flat, W_fc1), b_fc1))

# Fully connected Layer2
W_fc2 = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.001))
b_fc2 = tf.Variable(tf.truncated_normal([2048]))
#h_mat2 = tf.matmul(h_fc1, W_fc2)
#h_fc2b = tf.nn.bias_add(h_mat2, b_fc2)
#h_fc2 = tf.nn.relu(h_fc2b)
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))

# Output Layer
W_out = tf.Variable(tf.truncated_normal([2048, NUM_LABELS], stddev=0.001))
b_out = tf.Variable(tf.truncated_normal([NUM_LABELS]))
out_mat = tf.matmul(h_fc2, W_out)
#out_b = tf.nn.bias_add(out_mat, b_out)
#out = tf.(out_b)
#out = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_out), b_out))
out = tf.sigmoid(out_mat + b_out)

cost = tf.reduce_mean(tf.squared_difference(out, y_))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(out, y_)))
#rmse = tf.sqrt(tf.reduce_mean(out - y_))

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Initialize
init = tf.initialize_all_variables()
epochs=5

with tf.Session() as sess:
    sess.run(init)
    
    epoch=1
    while epoch<=epochs:
        print 'Training Epoch : {}'.format(epoch)
	dh = DataHelper(batch_size, test_idx=test_split)
        step = 1

        # Looks like training iters in the number of images to process
        while step * batch_size < test_split:
            # TODO get data in proper format
            batch_xs, batch_ys = dh.get_next_batch()
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
	    
            if step % display_step == 0:
                acc = sess.run(rmse, feed_dict={x: batch_xs, y_: batch_ys})
		loss = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys})
		
                print "Iter " + str(step * batch_size) + \
                      ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training RMSE= " + "{:.5f}".format(acc)
            step += 1

        print "Epoch {} finished".format(epoch)
	print "Testing Error:"
        test_data, test_labels = dh.get_test_data(test_size)
	test_step = 1
	_rmse=0.0
	test_rmse=0.0
	while test_step * batch_size < test_size:
	    start = (test_step - 1) * batch_size
	    end = test_step*batch_size
            tst_x = test_data[start:end]
	    tst_y = test_labels[start:end]
           
	    _rmse += sess.run(rmse, feed_dict={x: tst_x, y_: tst_y})
	    _rmse = pow(_rmse,2) * batch_size
	    test_rmse += _rmse
	    test_step += 1
	test_rmse = sqrt(test_rmse / float(test_size))
	print 'Average test rmse: {}'.format(test_rmse)
#	print 'Time taken in last epoch: {}'.format(int(time.time()-start1))
	epoch+=1
	
end = time.time()
print 'Average time taken per epoch: {}'.format((end-start1)/epochs)

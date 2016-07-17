import tensorflow as tf
import cPickle
import numpy as np


def unpickle(file):
	fo = open(file, 'rb')
	dicto = cPickle.load(fo)
	fo.close()
	return dicto


def weight_variable(shape, std = 1e-4):
  initial = tf.truncated_normal(shape, stddev = 1e-4)
  return tf.Variable(initial)


def bias_variable(shape, bias = 0.1):
  initial = tf.constant(bias, shape = shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def extract(n):
	x = []	
	y = []
	for i in range(n):
		data_labels = unpickle("CIFAR10_data/data_batch_" + str(i+1))
		p,q = data_labels['data'], data_labels['labels']
		x.extend(p)
		y.extend(q)
	# Seperate into R,G,B channels (row-major)
	for i in range(len(x)):
		x[i] = np.reshape(x[i], (3,32,32))
	return x,y


def next_batch(x, y, size, counter = 0):
	counter += size
	return [x[counter-size:counter],y[counter-size:counter]]


def train(X,Y, num_classes):
	sess = tf.Session()
	# Convolution
	x = tf.placeholder(tf.float32, shape = [3, 32, 32])
	y_ = tf.placeholder(tf.float32, shape = [None, 10])
	W_conv1 = weight_variable([5, 5, 3, 64])
	b_conv1 = bias_variable([64],0)
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
	# Pool
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	# Normalize
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	# Convolution
	W_conv2 = weight_variable([5, 5, 64, 64])
	b_conv2 = bias_variable([64],0)
	h_conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
	# Normalize
	norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
	# Pool
	h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	# FC layer
	W_fc1 = weight_variable([8 * 8 * 64, 384], 0.04)
	b_fc1 = bias_variable([384])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	# FC layer
	W_fc2 = weight_variable([384, 192], 0.04)
	b_fc2 = bias_variable([192])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	# Softmax (readout)
	W_fc3 = weight_variable([192, num_classes], 1/192.0)
	b_fc3 = bias_variable([num_classes])
	y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

	# Train network
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	
	counter = 0
	for i in range(250):
	  batch = train.next_batch(X,Y,200,counter)
	  counter += 200
	  if i%400 == 0:
	    train_accuracy = accuracy.eval(feed_dict={
	        x:batch[0], y_: batch[1], keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
	# print("test accuracy %g"%accuracy.eval(feed_dict={
	    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
	x,y = extract(5)	
	train(x,y,len(unpickle("CIFAR10_data/batches.meta")['label_names']))

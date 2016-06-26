import tensorflow as tf
import XOR

def train_and_evaluate():
	# Prepare data
	XOR.read()
	# Placeholder for input data
	x = tf.placeholder(tf.float32, [None, 32])
	# First (and only) layer
	W = tf.Variable(tf.zeros([32, 2]))
	# Bias
	b = tf.Variable(tf.zeros([2]))
	# Calculate softmax score
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	# Placeholder for actual label
	y_ = tf.placeholder(tf.float32, [None, 2])
	# Loss function: cross entropy	
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	# Gradient minimize: gradient descent
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	# Give life to nodes
	sess.run(init)
	# Train in 800 epochs, each with a batch size of 100
	for i in range(800):
	  batch_xs, batch_ys = XOR.next_batch(100)
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	# Calculate accuracy
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	test_data, test_label = XOR.test_model()
	x = (sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))
	# Return calculated accuracy
	return x 


if __name__ == "__main__":
	# Train and test model
	accuracy = train_and_evaluate()
	# Print accuracy of trained model
	print(accuracy)

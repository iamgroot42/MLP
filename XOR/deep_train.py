import tensorflow as tf
import XOR


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def train_and_evaluate():
	# Create interactive session
	sess = tf.InteractiveSession()
	# Prepare data
	XOR.read()
	# Placeholder for input data
	x = tf.placeholder(tf.float32, [None, 32])
	# Placeholder for actual label
	y_ = tf.placeholder(tf.float32, [None, 2])
	# layer-1
	W_fc1 = weight_variable([32, 16])
	b_fc1 = bias_variable([16])
	h_fc1 = tf.nn.softmax(tf.matmul(x, W_fc1) + b_fc1)
	# layer-2
	W_fc2 = weight_variable([16, 8])
	b_fc2 = bias_variable([8])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	# layer-3
	W_fc3 = weight_variable([8, 4])
	b_fc3 = bias_variable([4])
	h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
	# Dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
	# Readout layer
	W_fc4 = weight_variable([4, 2])
	b_fc4 = bias_variable([2])
	y_conv = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
	# Train network
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	# Calculate accuracy
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# Give life to nodes
	sess.run(tf.initialize_all_variables())
	# Train in 4000 epochs, each with a batch size of 200
	for i in range(4000):
		batch_xs, batch_ys = XOR.next_batch(200)
		if i%50 == 0:
			train_accuracy = accuracy.eval(feed_dict={
			x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	# Test model
	test_data, test_label = XOR.test_model()
	x = accuracy.eval(feed_dict={
		x: test_data, y_: test_label, keep_prob: 1.0})
	# Return calculated accuracy
	return x


if __name__ == "__main__":
	# Train and test model
	accuracy = train_and_evaluate()
	# Print accuracy of trained model
	print(accuracy)

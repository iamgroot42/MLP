from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Read training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Flattened 28*28 image as input
x = tf.placeholder(tf.float32, [None, 784])

# Weights matrix
W = tf.Variable(tf.zeros([784, 10]))

# Bias
b = tf.Variable(tf.zeros([10]))

# Output layer
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Actual labels for data
y_ = tf.placeholder(tf.float32, [None, 10])

# Error to be minimized
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Gradient for back-propogation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Run 1000 epochs, running randomly-sampled batches of size 100 every time
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#Calculate and print accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

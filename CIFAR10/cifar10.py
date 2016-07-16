# import tensorflow as tf
import cPickle
import numpy as np

def unpickle(file):
	fo = open(file, 'rb')
	dicto = cPickle.load(fo)
	fo.close()
	return dicto


def extract(n):
	x = []
	y = []
	# mapping = unpickle("CIFAR10_data/batches.meta")['label_names']
	for i in range(n):
		data_labels = unpickle("CIFAR10_data/data_batch_" + str(i+1))
		p,q = data_labels['data'], data_labels['labels']
		x.extend(p)
		y.extend(q)
	# Seperate into R,G,B channels
	for i in range(len(x)):
		x[i] = np.reshape(x[i], (1024,3))
	return x,y


def train(x,y):
	sess = tf.Session()
	x = tf.placeholder(tf.float32, shape=[None, 1024, 3])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])
	

if __name__ == "__main__":
	extract(5)	
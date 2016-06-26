import numpy as np
import os

# 80-20 traing:test ratio; no validation set
TRAIN_RATIO = 0.8
TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []
BATCH_COUNT = 0
TRAIN_COUNT = 0

# 1-hot representation
def one_hot(x):
	if x == 1:
		return [0,1]
	else:
		return [1,0]

# Generate train,test data and write to file (only if required)
def populate(n = 32,s = 1000000):
	should = False
	if not os.path.isdir("XOR"):
		os.mkdir("XOR")
		should = True
	else:
		if not (os.path.exists("XOR/input") and os.path.exists("XOR/labels")):
			if os.path.exists("XOR/input"):
				os.remove("XOR/input")
			if os.path.exists("XOR/labels"):
				os.remove("XOR/labels")
			should = True

	if should:
		f = open("XOR/input",'w')
		g = open("XOR/labels",'w')
		for i in range(s):
			print (100*(i+1))/float(s),"%"
			x = np.random.choice([1,0], n)
			for y in x:
				f.write(str(y) + ",")
			f.write("\n")
			label = 0
			for i in x:
				label = label ^ i
			g.write(str(label) + "\n")

# Read data from file and load into memory
def read():
	global TRAIN_DATA
	global TRAIN_LABELS
	global TRAIN_COUNT
	global TEST_DATA
	global TEST_LABELS
	global TRAIN_RATIO
	if not (os.path.exists("XOR/input") and os.path.exists("XOR/labels")):
		populate()
	f = open("XOR/input",'r')
	g = open("XOR/labels",'r')
	all_data = []
	all_labels = []
	num_ex = 0
	for x in f:
		row = x.rstrip('\n').split(',')[:-1]
		row = [int(x) for x in row]
		all_data.append(row)
		num_ex += 1
	for x in g:
		all_labels.append(one_hot(int(x.rstrip('\n'))))

	TRAIN_COUNT = int(TRAIN_RATIO*num_ex)
	TRAIN_DATA = all_data[:TRAIN_COUNT]
	TRAIN_LABELS = all_labels[:TRAIN_COUNT]
	TEST_DATA = all_data[TRAIN_COUNT:]
	TEST_LABELS = all_data[TRAIN_COUNT:]

# Get the next required training-batch of 'n' size
def next_batch(n):
	global BATCH_COUNT
	global TRAIN_DATA
	global TRAIN_LABELS
	if BATCH_COUNT + n <= TRAIN_COUNT:
		bc = BATCH_COUNT
		BATCH_COUNT += n
		return TRAIN_DATA[bc:bc+n], TRAIN_LABELS[bc:bc+n]
	return []

# Test the given model
def test_model():
	return TRAIN_DATA, TRAIN_LABELS

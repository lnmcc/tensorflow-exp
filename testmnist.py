import tensorflow as tf
import os
import gzip
import tempfile
import shutil
from six.moves import urllib
from tensorflow.examples.tutorials.mnist import input_data

def download(directory, filename):
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading %s to %s ' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
        tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


# Download MNIST Datesets
mnist_director = '/Users/sjie/Projects/tensorflow/proj/mnist/'
#mnist_data_set = download(mnist_director, 'train-images-idx3-ubyte')
#print("Training data: ", mnist_data_set)
mnist = input_data.read_data_sets(mnist_director, one_hot=True)
print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)
print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])

#batch_size = 100
#xs, ys = mnist.train.next_batch(batch_size)
#print('xs = {0}\n  ys = {1}'.format(xs, ys))
#print("X shape:", xs.shape)
#print("Y shape:", ys.shape)

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_SETPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weight2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)))
        avg_class.average(biases1)
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
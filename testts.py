from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

##for i in range(100):
##    _, loss_value = sess.run((train, loss))
##    print(loss_value)
my_image = tf.zeros([10, 299, 299, 3])
r = tf.rank(my_image)
print(sess.run(my_image))
print(sess.run(r))
print(my_image.shape[1])

rank_three_tensor = tf.ones([3, 4, 5])
print(sess.run(rank_three_tensor))
matrix = tf.reshape(rank_three_tensor, [6, 10])
print(sess.run(matrix))
matrixB = tf.reshape(matrix, [3, -1])
print(sess.run(matrixB))
matrixAlt = tf.reshape(matrixB, [4, 3, -1])
print(sess.run(matrixAlt))

constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval(session=sess))



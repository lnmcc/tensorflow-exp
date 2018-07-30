import tensorflow as tf

a = tf.constant(5)
sess = tf.Session()

ones = tf.ones([6])
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)

with sess.as_default():
    print(a.eval())
    print(ones.eval())
    print(uniform.eval())
    print(normal.eval())

sess.close()


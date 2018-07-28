import tensorflow as tf

a = tf.constant(5)
sess = tf.Session()

with sess.as_default():
    print(a.eval())

sess.close()
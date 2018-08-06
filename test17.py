import tensorflow as tf

t = tf.constant([[0, 1, 2], [3, 4, 5]])

with tf.Session() as sess:
    print(sess.run(tf.shape(t)))
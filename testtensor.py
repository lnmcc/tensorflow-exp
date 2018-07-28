import tensorflow as tf

a = tf.constant([5, 3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(c, b, name="add_d")

with tf.Session() as sess:
    print(sess.run(d))

import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[None], name="input_placehplder_a")
b = tf.reduce_prod(a, name="product_b")

sess = tf.Session()
feed_dict = {a: [2, 8, 2]}
print(sess.run(b, feed_dict=feed_dict))
import tensorflow as tf

a = tf.add(2, 5)
b = tf.multiply(a, 3)

sess = tf.Session()
print(sess.run(b))

replace_dict = {a: 15}
ret = sess.run(b, feed_dict=replace_dict)
print(ret)
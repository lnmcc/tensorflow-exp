import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
init_op = tf.global_variables_initializer ()

print(result.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "/tmp/tf/model.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "/tmp/tf/model.ckpt")
    print(sess.run(result))

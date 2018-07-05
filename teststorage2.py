import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.all_variables():
    print(variables.name)

print("-" * 20)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.all_variables())

for variables in tf.all_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    saver.save(sess, "/tmp/tf/model.ckpt")
    print(sess.run([v, ema.average(v)]))

print('-' * 20)
v = tf.Variable(0, dtype=tf.float32, name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "/tmp/tf/model.ckpt")
    print(sess.run(v))

v = tf.Variable(0, dtype=tf.float32, name="v")
print(ema.variables_to_restore())
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "/tmp/tf/model.ckpt")
    print(sess.run(v))

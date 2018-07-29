import tensorflow as tf

a = tf.constant(5)

trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)

my_var = tf.Variable(1)
my_var_times_two = my_var.assign(my_var * 2)
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
print(sess.run(my_var_times_two))
print(sess.run(my_var_times_two))
print(sess.run(my_var_times_two))

sess.close()

sess = tf.Session()
sess.run(init)
print(sess.run(my_var_times_two))

#with sess.as_default():
#    print(a.eval())
#    print(trunc.eval())
    


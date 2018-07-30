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

<<<<<<< HEAD
ones = tf.ones([6])
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)

with sess.as_default():
    print(a.eval())
    print(ones.eval())
    print(uniform.eval())
    print(normal.eval())

sess.close()
=======
#with sess.as_default():
#    print(a.eval())
#    print(trunc.eval())
    
>>>>>>> 319b91e917fec90f9385d93d74060da3fd55c557


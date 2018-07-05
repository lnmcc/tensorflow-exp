from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

my_variable = tf.get_variable("my_variable", [1, 2, 3])
my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])
tf.add_to_collection("my_collection_name", my_local)
print(tf.get_collection("my_collection_name"))

sess = tf.Session()
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
#w = tf.get_variable("w", initializer=v.initializer_value() + 1)
w = v + 1

assignment = v.assign_add(1)
tf.global_variables_initializer().run()
sess.run(assignment)

c_0 = tf.constant(0, name="c")
c_1 = tf.constant(2, name="c")
with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")
    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c")
    c_4 = tf.constant(4, name="c")
    
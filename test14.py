import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("variables"):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        total_output = tf.Variable(0.0, dtype=tf.int32, trainable=False, name="total_outpu")

    with tf.name_scope("transformation"):
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None], name="input_placehplder_a")

        with tf.name_scope("intermedia_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")

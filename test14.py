import tensorflow as tf
import os

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("variables"):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    with tf.name_scope("transformation"):
        with tf.name_scope("input"):
            a = tf.placeholder(tf.float32, shape=[None], name="input_placehplder_a")

        with tf.name_scope("intermedia_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")

        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")

        with tf.name_scope("update"):
            update_total = total_output.assign_add(output)
            increment_step = global_step.assign_add(1)

        with tf.name_scope("summaries"):
            avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
            tf.summary.scalar("total_summary", update_total)
            tf.summary.scalar("average_summary", avg)

        with tf.name_scope("global_ops"):
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            merged_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./improved_graph', graph)
sess.run(init_g)
# sess.run(init_l)

def read_cvs(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + 
                        "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    decorded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decorded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    _, step, summary, br = sess.run([output, increment_step, merged_summaries, b],
                    feed_dict=feed_dict)
    print("b = %d" % br)
    writer.add_summary(summary, global_step=step)

run_graph([2, 8])
run_graph([3, 1, 3, 1])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

writer.flush()
writer.close()
sess.close()
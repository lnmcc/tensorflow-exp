import tensorflow as tf
import os

W = tf.Variable(tf.zeros([4, 3]), name="weights")
b = tf.Variable(tf.zeros([3], name="bias"))

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    return tf.nn.softmax(combine_inputs(X))

def loss(X, Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=combine_inputs(X), logits=Y))

def read_cvs(batch_size, file_name, record_defaults):
    print("read_csv(), file_name: %s" % file_name)
    filename_queue = tf.train.string_input_producer([file_name])
    print("filename_queue: %s" % file_name)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    decorded = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decorded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)

def inputs():
    print("inputs()")
    sepal_length, speal_width, petal_length, petal_width, label = \
        read_cvs(1, "./datasets/iris.data",
            [[0.0], [0.0], [0.0], [0.0], [""]])

    #print("label:", sess.run(label))

    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))

    features = tf.transpose(tf.stack([sepal_length, speal_width, petal_length, petal_width]))
    return features, label_number, label

def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X, Y, label = inputs()
    #total_loss = loss(X, Y)
    #train_op = train(total_loss)

    #sepal_length, speal_width, petal_length, petal_width, label = \
        #read_cvs(100, "./datasets/iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_steps = 10000

    print("Label:", sess.run(label))
    print("X: ", sess.run(X))
    print("Y: ", sess.run(Y))
    #for step in range(training_steps):
        #sess.run([train_op])

        #if step % 100 == 0:
           #print("loss: ", sess.run([total_loss]))

    #print(sess.run(passenger_id))
    #print(sess.run(survived))

    #evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads) 

    #writer = tf.summary.FileWriter('./board', graph=tf.get_default_graph())
    #writer.flush()
    #writer.close()

    sess.close()

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/Users/sjie/Projects/tensorflow/proj/tensorflow-exp/cat.jpg", 'r').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
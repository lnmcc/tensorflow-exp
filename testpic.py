# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/Users/sjie/Projects/tensorflow/proj/tensorflow-exp/cat2.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    plt.imshow(img_data.eval())
    #plt.show()
    #img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("/Users/sjie/Projects/tensorflow/proj/tensorflow-exp/cat2-1.jpg", 'wb') as f:
        f.write(encoded_image.eval())

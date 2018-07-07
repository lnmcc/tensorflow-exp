# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile("/Users/sjie/Projects/tensorflow/proj/tensorflow-exp/cat2.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.resize_images(img_data, [180, 267], method=1)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)

    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    #print(result.eval())
    #print(img_data.eval())
    plt.imshow(result.eval())
    plt.show()
    #img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    #encoded_image = tf.image.encode_jpeg(img_data)
    #with tf.gfile.GFile("/Users/sjie/Projects/tensorflow/proj/tensorflow-exp/cat2-1.jpg", 'wb') as f:
    #    f.write(encoded_image.eval())

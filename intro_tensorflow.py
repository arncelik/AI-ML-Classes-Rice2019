import tensorflow as tf
import numpy as np
import matplotlib as plt

dtype = tf.float32

test_constant = tf.constant(10.0, dtype=tf.float32)
add_one_operation = test_constant + 1

with tf.Session() as sess:
    print(sess.run(add_one_operation))
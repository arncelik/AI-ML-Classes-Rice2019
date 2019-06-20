import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
image_height = 28
image_width = 28

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

class ConvNet:

    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels],
                                          name="inputs")
        print(self.input_layer.shape)

        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same",
                                        activation=tf.nn.relu)
        print(conv_layer_1.shape)

        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
        print(pooling_layer_1.shape())
# Add a conv2d and pooling layer
        conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5], padding="same",
                                        activation=tf.nn.relu)
        print(conv_layer_2.shape)

        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)

        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)

        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)

        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)

        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())
steps = 5000
batch_size = 32

test_img = x_test[1]
plt.imshow(test_img)
plt.show()

test_img = test_img.reshape(-1, 28, 28, 1)


x_train = x_train.reshape(-1, height, image_width, 1)
cnn = CNN(28, 28, 1, 10)
with tf.Session() as sess:
    sess.run(tf.global_variables_intializer())
    sess.run(tf,local_variables_initializer())
    step = 0
    while step < steps:
        sess.run((cnn.train_operation, cnn.accuracy_opp), feed_dict = {cnn.input_layer:x_train[step:step+batch_size],
                                                                             cnn.labels:y_train[step]})
        step  += batch_size
        print(sess.run(cnn.choice, feed_dict={cnn.input_layer: test_img}))

test_image = x_test[1]
plt.imshow(test_img)
pllt.show()
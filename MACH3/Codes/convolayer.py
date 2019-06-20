import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.datasets as skd

image = skd.load_sample_image("china.jpg")
plt.imshow(image)
plt.show()

input_layer = tf.placeholder(dtype =tf.float32, shape = [None, 427, 640, 3])
conv_1 = tf.layers.conv2d(input_layer, filters=3, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)

init = tf.global_variables_initializer()
with tf.Session() as sess:
        sess.run(init)
        output = sess.run(conv_1, feed_dict = {input_layer:[image]})
        print(output[0].shape)

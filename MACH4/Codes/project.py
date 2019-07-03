import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt

image_index = 40000
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap = "Greys")
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)


model.evaluate(x_test, y_test)

image_index = 3000

plt.imshow(x_test[image_index].reshape(28,28),cmap='Greys')

pred = model.predict(x_test[image_index].reshape(1, 28,28,1))

print(pred.argmax())

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist_data = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist_data.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', #how will we calculate the error to minimize the loss
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)

model.save(r'C:\Digit Recognition\digit_recognition_model.model')

new_model=tf.keras.models.load_model(r'C:\Digit Recognition\digit_recognition_model.model')

predictions = new_model.predict(x_test)

plt.imshow(x_test[110])
np.argmax(predictions[110])

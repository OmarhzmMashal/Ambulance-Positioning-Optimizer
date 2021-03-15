import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle 
from PIL import Image, ImageOps

letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
           'Q','R','S','T','U','V','W','X','Y','Z']


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(THIS_FOLDER, 'sign_mnist_train.csv')
test_csv = os.path.join(THIS_FOLDER, 'sign_mnist_test.csv')
test_imgpath = os.path.join(THIS_FOLDER, 'n1.jpg')
modelsave =  os.path.join(THIS_FOLDER, 'modelSignV2.h5')


def get_data(filename):
    with open(filename) as training_file:
      # Your code starts here
        my_arr = np.loadtxt(filename, delimiter=',', skiprows=1)
        # get label & image arrays
        labels = my_arr[:,0].astype('int')
        images = my_arr[:,1:]
        # reshape image from 784 to (28, 28)
        images = images.astype('float').reshape(images.shape[0], 28, 28)
        # just in case to avoid memory problem
        my_arr = None
      # Your code ends here
    return images, labels

train_images, train_labels = get_data(train_csv)
test_images, test_labels = get_data(test_csv)

print(train_labels.shape)

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(75, (3, 3),  strides = 1, padding = 'same', activation='relu', input_shape=(28, 28, 1)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D((2, 2),strides = 1 , padding = 'same'),
    
tf.keras.layers.Conv2D(50, (3, 3), strides = 1 , padding = 'same', activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D((2,2) , strides = 1 , padding = 'same'),
    
tf.keras.layers.Conv2D(25, (3, 3), strides = 1 , padding = 'same', activation='relu'),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D((2,2) , strides = 1 , padding = 'same'),
    
tf.keras.layers.Flatten(),

tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.Dense(25, activation='softmax')
])
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])

train_images = train_images.reshape((-1,28,28, 1))
test_images = test_images.reshape((-1,28,28, 1))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    fill_mode='nearest',
    horizontal_flip=True
)
train_gen = train_datagen.flow(train_images, train_labels)


val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow(test_images, test_labels)


history = model.fit_generator(train_gen, epochs=100, verbose=1, validation_data=val_gen)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test_imgpath = os.path.join(THIS_FOLDER, '1.jpg')
image = Image.open(test_imgpath)
image = image.resize((28,28))
image = ImageOps.grayscale(image)
image = np.array(image).reshape((-1,28,28,1))
image = image/255.0
predicted_int = np.argmax(model.predict(image))
print(letters[predicted_int])

model.save(modelsave)





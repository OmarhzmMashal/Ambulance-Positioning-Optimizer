import os
import sys
import face_recognition
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import glob
import ntpath
from pathlib import Path
import socket
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tensorflow.keras import initializers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print(len(tf.config.experimental.list_physical_devices('GPU')))

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# read data
data = pd.read_csv(THIS_FOLDER + "/data.csv")

# folder variables for saving
modelsave = os.path.join(THIS_FOLDER, 'emModelV5.h5')
modelweights = os.path.join(THIS_FOLDER, 'emWeightsV5.h5')

#choose only happy, sad, and neutral
lbl_index = [3,4,6]
data = data[data.emotion.isin(lbl_index)]

label = data['emotion']
img = data['pixels']  

# create an array numbers representing images from raw data
img = img.apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('float32'))
img = np.stack(img, axis=0)

# encode 
le = LabelEncoder()
label = le.fit_transform(label)
label = to_categorical(label) #one hot encoding

X_train, X_valid, y_train, y_valid = train_test_split(img, label,
                                                      shuffle=True,
                                                      stratify=label,
                                                      test_size=0.1)

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(256, (5, 5),  strides = 1, activation='relu', 
                       input_shape=(48, 48, 1),
                        kernel_initializer='he_normal'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.AveragePooling2D((2, 2),strides = 1),

tf.keras.layers.Conv2D(256, (5, 5), strides = 1 , activation='relu',
                       kernel_initializer='he_normal'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.AveragePooling2D((2,2) , strides = 1),

tf.keras.layers.Conv2D(128, (3, 3), strides = 1 , activation='relu',
                       kernel_initializer='he_normal'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.AveragePooling2D((2,2) , strides = 1),

tf.keras.layers.Flatten(),

tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss = 'categorical_crossentropy', 
              metrics=['acc'])

X_train = X_train.reshape((-1,48,48, 1))
X_valid = X_valid.reshape((-1,48,48, 1))

augmantation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True
)
train_data = augmantation.flow(X_train, y_train, batch_size=32)

augmantationt = ImageDataGenerator(rescale=1./255)

test_data = augmantationt.flow(X_valid, y_valid)

cb = EarlyStopping(
    monitor='acc',
    min_delta=0.005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)


history = model.fit_generator(train_data, epochs=100, verbose=1, 
                              validation_data=test_data,  callbacks=cb)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save(modelsave)
model.save_weights(modelweights)











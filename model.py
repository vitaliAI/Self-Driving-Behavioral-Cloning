import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math


from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2

import json
import random


data = pd.read_csv('data/new_data/new_log.csv',
                   dtype={'center': str,'steering': np.float32},
                   skipinitialspace=1)
print(data.dtypes)

X_train = data['center']
y_train = data['steering']

print('Training data size = ', len(X_train))
print('Training labels size = ',len(y_train))



# split data into train, validate and test data
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.3,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.2,random_state=42)
print('Data splitted into train, validation anda test data')
print('Training data size = ', len(X_train))
print('Training labels size = ',len(y_train))
print('Validation data size = ',len(X_val))
print('Validation labels size = ',len(y_val))
print('Test data size = ',len(X_test))
print('Test labels size = ',len(y_test))



# define the model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(80, 300, 3),output_shape=(80, 300, 3)))
model.add(Conv2D(24, (5, 5),strides=(2, 2),kernel_regularizer='l2',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Conv2D(36, (5, 5),strides=(2, 2),kernel_regularizer='l2',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5),strides=(2, 2),kernel_regularizer='l2',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),strides=(1, 1),kernel_regularizer='l2',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),strides=(1, 1),kernel_regularizer='l2',dim_ordering='tf'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100, name='fc1'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50, name='fc2'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10, name='fc3'))
model.add(Activation('relu'))
model.add(Dense(1,name='output'))

# for a mean squared error regression problem
model.compile(optimizer=Adam(lr=0.0001),loss='mean_squared_error')
model.summary()
print('Model defined')

from PIL import Image


def generator(X, Y):
    while 1:
        for (x, y) in zip(X, Y):
            path, angle = (x, y)
            angle = np.reshape(angle, [1])
            image = Image.open('./data/new_data/' + path)
            image_array = np.asarray(image)
            transformed_image_array = image_array[None, :, :, :]
            yield transformed_image_array, angle

#train the model
print('')
iterations = 30
train_batch_size = 100
val_batch_size = 100

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, monitor = 'val_loss', save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit_generator(generator(X_train, y_train),
                    samples_per_epoch= train_batch_size,
                    nb_epoch=iterations,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_data=generator(X_val, y_val),
                    nb_val_samples=val_batch_size)

print('----------------- Model trained! -----------------')

model.save('model.h5')

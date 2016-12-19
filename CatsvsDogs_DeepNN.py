import os
import re
import csv
import sys
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2

from Keras_Functions import save_model_to_disk, load_model_from_disk, img_to_matrix

# For reproducibility
np.random.seed(1337)

# setting training parameters
batch_size = 50
nb_classes = 2
nb_epoch = 400

# Flag if the model should be loaded or trained
load_model_flag = True
if '-train' in sys.argv:
    load_model_flag = False

# Set resizing image dimensions
img_rows, img_cols = 100, 100

# Set neural net's input shape
input_shape = (img_rows, img_cols, 1)

# Get training and validation data
img_dir = 'train/'
image_files = [img_dir + f for f in os.listdir(img_dir)]
shuffle(image_files)
train_size = 9 * len(image_files) // 10
train_images = image_files[:train_size]
test_images = image_files[train_size:]

X_train = [img_to_matrix(train_image, img_rows, img_cols) for train_image in train_images]
Y_train = [0 if 'cat' in f.split('/')[-1] else 1 for f in train_images]

print('Loaded training data')

X_test = [img_to_matrix(test_image, img_rows, img_cols) for test_image in test_images]
Y_test = [0 if 'cat' in f.split('/')[-1] else 1 for f in test_images]

print('Loaded testing data')

# Convert data into numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Normalize the pixels to be in the range of [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Cocnvert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Either train new model or load old one
if load_model_flag:
    model = load_model_from_disk()
else:
    print("Setting up model")
    model = Sequential()

    model.add(Convolution2D(8, 9, 9, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 7, 7))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(250, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))


    model.add(Dense(nb_classes, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer= sgd, metrics=['accuracy'])

    print("Setting up image generator")

    # Create data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

    print("Fitting image generator")

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    print("Creating early stopping callback")

    early_stopping = EarlyStopping(monitor='loss', min_delta=1e-3, patience=3, mode='auto')

    print("Fitting model")

    # fits the model on batches with real-time data augmentation
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), 
        samples_per_epoch= len(X_train),
        nb_epoch=nb_epoch,
        verbose=1,
        #callbacks = [early_stopping],
        validation_data=(X_test, Y_test))

    save_model_to_disk(model)

# Test our model
print("Evaluating model")
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])    

# Get prediction data
img_dir = 'test/'
predict_images = [img_dir + f for f in os.listdir(img_dir)]
ids = [f.split('/')[-1][:-4] for f in predict_images]

X_predict = [img_to_matrix(test_image, img_rows, img_cols) for test_image in predict_images]
X_predict = np.array(X_predict)

print('Loaded prediction data')

# Normalize pixels to range of [0, 1]
X_predict = X_predict.astype('float32')
X_predict /= 255

predictions = model.predict(X_predict)

# Write predictions
with open('predictions.csv', 'w', newline='') as fp:
    csv_writer = csv.writer(fp, delimiter=',')
    csv_writer.writerow(['id', 'label'])
    csv_writer.writerows(zip(ids, predictions[:, 1]))

print('Saved predictions to disk')
import os
import re
import csv
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from Keras_Functions import save_model, load_model, img_to_matrix

# For reproducibility
np.random.seed(1337)

# setting training parameters
batch_size = 200
nb_classes = 2
nb_epoch = 30

# Flag if the model should be loaded or trained
load_model_flag = True

# Set resizing image dimensions
img_rows, img_cols = 100, 100

# Set neural net's input shape
input_shape = (img_rows, img_cols, 1)

# Get training and validation data
img_dir = "train/"
image_files = [img_dir + f for f in os.listdir(img_dir)]
shuffle(image_files)
train_size = 5 * len(image_files) // 6
train_images = image_files[:train_size]
test_images = image_files[train_size:]

X_train = [img_to_matrix(train_image, img_rows, img_cols) for train_image in train_images]
Y_train = [0 if "cat" in f.split('/')[-1] else 1 for f in train_images]

print("Loaded training data")

X_test = [img_to_matrix(test_image, img_rows, img_cols) for test_image in test_images]
Y_test = [0 if "cat" in f.split('/')[-1] else 1 for f in test_images]

print("Loaded testing data")

# Convert data into numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Normalize the pixels to be in the range of [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Cocnvert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Either train new model or load old one
if load_model_flag:
    model = load_model()
else:
    model = Sequential()

    model.add(Convolution2D(8, 9, 9, input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 7, 7))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(250, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    save_model(model)

# Test our model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])    

# Get prediction data
img_dir = "test/"
predict_images = [img_dir + f for f in os.listdir(img_dir)]
ids = [f.split('/')[-1][:-4] for f in predict_images]

X_predict = [img_to_matrix(test_image, img_rows, img_cols) for test_image in predict_images]

print("Loaded prediction data")

X_predict = np.array(X_predict)

# Normalize pixels to range of [0, 1]
X_predict = X_predict.astype('float32')
X_predict /= 255

predictions = model.predict(X_predict)

# Write predictions
with open('predictions.csv', 'w', newline='') as fp:
    csv_writer = csv.writer(fp, delimiter=',')
    csv_writer.writerow(['id', 'label'])
    csv_writer.writerows(zip(ids, predictions[:, 1]))
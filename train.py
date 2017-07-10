#!/usr/bin/env python

import check_env

NUMBER_OF_EPOCHS = 10

from img_lib import display_images_and_labels, load_data, normalize

import warnings
warnings.filterwarnings('ignore')

import os

print ("Loading data")
# Load datasets.
ROOT_PATH = "./data"
# data_dir = os.path.join(ROOT_PATH, "speed-limit-signs")
data_dir = os.path.join(ROOT_PATH, "augmented-signs")

images, labels = load_data(data_dir, type=".png")
# display_images_and_labels(images, labels)

test_data_dir = os.path.join(ROOT_PATH, "speed-limit-signs")
images_test, labels_test= load_data(test_data_dir, type=".ppm")

print ("Normalizing")

# X,y = normalize(images, labels)
# X_test, y_test = normalize(images_test, labels_test)
# unaugmented data
X, y = normalize(images_test, labels_test)

print ("Setting up VGG style model")

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D

drop_out = 0.5
# drop_out = 0.25
# drop_out = 0.0

# input tensor for a 3-channel 64x64 image
inputs = Input(shape=(64, 64, 3))

# one block of convolutional layers

# 32 filters with a 3x3 kernel, outputs 64x64x32 tensor
x = Convolution2D(32, 3, 3, activation='relu')(inputs)
x = Convolution2D(32, 3, 3, activation='relu')(x)
x = Convolution2D(32, 3, 3, activation='relu')(x)

# max pooling with 2x2 window, reducing data to a fourth, reduces risk of overfitting
x = MaxPooling2D(pool_size=(2, 2))(x)

# drops 25% / 50% of all nodes at training (but not for test/prediction), also reduces risk of overfitting
x = Dropout(drop_out)(x)
# http://cs231n.github.io/neural-networks-2/#reg

# one more block
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(drop_out)(x)

# one more block
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(drop_out)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(drop_out)(x)

# softmax activation, 6 categories
predictions = Dense(6, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
import keras

checkpoint_callback = keras.callbacks.ModelCheckpoint('./tmp/model-checkpoints/weights.epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5');

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

tb_callback = keras.callbacks.TensorBoard(log_dir='./tmp/tf_log')

print ("Ready for training")

# We can use all our data for training, because we have a completely separate set for testing later
# X_train, y_train = X, y

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=11)

from datetime import tzinfo, timedelta, datetime

print(datetime.utcnow().isoformat())
# model.fit(X_train, y_train, epochs=NUMBER_OF_EPOCHS, batch_size=500, validation_split=0.3,
#           callbacks=[tb_callback, checkpoint_callback, early_stopping_callback])
model.fit(X_train, y_train, epochs=100, batch_size=10)
print(datetime.utcnow().isoformat())

print ("Training loss / accuracy ")
train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=200)
print (train_loss, train_accuracy)

test_data_dir = os.path.join(ROOT_PATH, "speed-limit-signs")
images, labels= load_data(test_data_dir, type=".ppm")
X_test, y_test = normalize(images, labels)
print ("Test loss / accuracy (on unaugmented data)")
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=200)
print (test_loss, test_accuracy)

model.save('./tmp/models/conv-vgg-augmented.hdf5')

from distutils.version import StrictVersion

import sklearn
assert StrictVersion(sklearn.__version__ ) >= StrictVersion('0.18.1')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
assert StrictVersion(tf.__version__) >= StrictVersion('1.0.0')
print(tf.__version__)

import keras
assert StrictVersion(keras.__version__) >= StrictVersion('2.0.0')
print(keras.__version__)

import numpy as np
from keras.models import load_model

model = load_model('models/conv-vgg.hdf5')
# model = load_model('models/conv-vgg.h5')
# model = load_model('models/conv-residual.h5')

import skimage.data
import skimage.transform

def transform_to_image(file):
    # turn into image
    image = skimage.data.imread(file)
    print(image.shape)
    # Resize images
    image64 = skimage.transform.resize(image, (64, 64))
    print(image64.shape)
    return image64


def predict(image):
    # normalize
    X_sample = np.array([image])
    prediction = model.predict(X_sample)
    predicted_category = np.argmax(prediction, axis=1)
    return predicted_category, prediction
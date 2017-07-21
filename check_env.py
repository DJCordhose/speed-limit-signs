#!/usr/bin/env python

from distutils.version import StrictVersion

print ("Checking Matplotlib")
import matplotlib
print (matplotlib.__version__ )
assert StrictVersion(matplotlib.__version__) >= StrictVersion('2.0.0')
print ("Matplotlib ok")

print ("Checking skimage")
import skimage
print (skimage.__version__ )
assert StrictVersion(skimage.__version__) >= StrictVersion('0.13.0')
print ("skimage ok")

print ("Checking Sklean")
import sklearn
print (sklearn.__version__ )
assert StrictVersion(sklearn.__version__ ) >= StrictVersion('0.18.1')
print ("Sklean ok")

print ("Checking TensorFlow")
import tensorflow as tf
print (tf.__version__ )
tf.logging.set_verbosity(tf.logging.ERROR)
assert StrictVersion(tf.__version__) >= StrictVersion('1.1.0')
print ("TensorFlow ok")

print ("Checking Keras")
import keras
print (keras.__version__ )
assert StrictVersion(keras.__version__) >= StrictVersion('2.0.0')
print ("Sklean ok")

print ("Checking Numpy")
import numpy
print (numpy.__version__ )
assert StrictVersion(numpy.__version__) >= StrictVersion('1.12.0')
print ("Numpy ok")


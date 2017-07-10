import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import skimage.data

def load_data(data_dir, type=".ppm"):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(type)]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    # plt.show()
    plt.savefig("./sample_images.png")


import numpy as np
import skimage.transform
from keras.utils.np_utils import to_categorical

def normalize(images, labels):
    # Resize images
    images64 = [skimage.transform.resize(image, (64, 64))
                for image in images]

    # for image in images64[:5]:
    #     print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
    #
    y = np.array(labels)
    X = np.array(images64)


    num_categories = 6

    y = to_categorical(y, num_categories)
    return X,y


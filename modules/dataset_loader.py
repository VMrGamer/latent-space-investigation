## @package modules
#  Helps in various part of the projects
#
#

import tensorflow as tf
import numpy as np
import cv2

## @title dataset_loader
#  Dataset loader class to automate dataset loading.
class dataset_loader:
    
    ## A class variable storing information on the data
    dataset_dict = {
        'mnist': [60000, 10000, 32, (28, 28, 1), 28, False, False],
        'cifar10': [60000, 10000, 32, (32, 32, 3), 32, True, False],
        'fashion_mnist': [50000, 10000, 32, (28, 28, 1), 28, False, False],

    }

    ## A class variable storing dataset retrieval objects from tensorflow.
    data_loader = {
        'mnist': tf.keras.datasets.mnist,
        'cifar10': tf.keras.datasets.cifar10,
        'fashion_mnist': tf.keras.datasets.fashion_mnist,
    }

    ## constructor
    def __init__(self, dataset, gray=True):
        self.train_size, self.test_size, self.batch_size, self.input_shape, self.digit_size, self.convert, self.pad = self.dataset_dict[dataset]
        if gray:
            (self.train_images, _), (self.test_images, _) = self.data_loader[dataset].load_data()

            if self.convert:
                self.train_images = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in self.train_images])
                self.test_images = np.array([cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in self.test_images])
                self.input_shape[2] = 1

            self.train_images = self.prep(self.train_images)
            self.test_images = self.prep(self.test_images)

            self.train_dataset = (tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.train_size).batch(self.batch_size))
            self.test_dataset = (tf.data.Dataset.from_tensor_slices(self.test_images).shuffle(self.test_size).batch(self.batch_size))

    ## A function to prepare images for input
    #  normalize, reshape and convert dtype to float32
    def prep(self, imgs):
        norm_img = imgs.reshape((imgs.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2])) / 255.
        return np.where(norm_img > .5, 1.0, 0.0).astype('float32')

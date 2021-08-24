## @package modules
#  Helps in various part of the projects
# few portions adapted from TensorFlow Tutorial, Convolutional Variational Autoencoders
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb
#
# LICENSE
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

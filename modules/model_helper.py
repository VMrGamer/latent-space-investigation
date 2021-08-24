## @package modules
# Model function helper
# Adapted from TensorFlow Tutorial, Convolutional Variational Autoencoders
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


import tensorflow as tf
import numpy as np
import cv2

import time
import glob

import tensorflow_probability as tfp
import PIL

## @title model_helper
#  Convolutional variational autoencoder class
class model_helper:
    ## constructor
    def __init__(self, model, data, epochs=10, latent_dimen=2, n_ex = 16, optimizer=None):
        self.model = model

        self.data = data

        self.epochs = epochs

        # latent space dimension
        self.latent_dimen = latent_dimen
        self.n_ex = n_ex

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        self.random_vector_for_generation = tf.random.normal(shape=[n_ex, latent_dimen])

        # initialize optimizer
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(1e-4)
        else:
            self.optimizer = optimizer

        # Pick a sample of the test set for generating output images
        assert data.batch_size >= n_ex
        for test_batch in data.test_dataset.take(1):
            self.test_sample = test_batch[0:n_ex, :, :, :]

    ## function to compute log norm
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    ## function to compute cross entropy loss
    def compute_loss(self, x):
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    ## function to perform training
    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    ## function to decode and plot latent space
    def plot_latent_images(self, n):
        """Plots n x n digit images decoded from the latent space."""

        norm = tfp.distributions.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
        grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
        image_width = data.digit_size*n
        image_height = image_width
        image = np.zeros((image_height, image_width))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                x_decoded = self.model.sample(z)
                digit = tf.reshape(x_decoded[0], (data.digit_size, data.digit_size))
                image[i * data.digit_size: (i + 1) * data.digit_size,
                        j * data.digit_size: (j + 1) * data.digit_size] = digit.numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='Greys_r')
        plt.axis('Off')
        plt.show()

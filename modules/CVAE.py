## @package modules
#  Convolutional variational autoencoder
#
#

import tensorflow as tf
import numpy as np


## @title CVAE
#  Convolutional variational autoencoder class
class CVAE(tf.keras.Model):

    def __init__(self, latent_dimen=2, inp):
        super(CVAE, self).__init__()
        self.latent_dimen = latent_dimen
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=inp),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dimen + latent_dimen),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dimen,)),
                tf.keras.layers.Dense(units=inp[0]*inp[1]*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(inp[0]//4, inp[1]//4, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dimen))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
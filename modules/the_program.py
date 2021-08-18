from CVAE import CVAE
from dataset_loader import dataset_loader
from model_helper import model_helper

from IPython import display

import tensorflow as tf
import numpy as np

import time
import glob
import matplotlib.pyplot as plt

class the_program:
    def __init__(self, dataset='mnist', latent_dimen=2):
        self.data = dataset_loader('mnist')
        self.model = CVAE(latent_dimen, self.data.input_shape)
        self.m_helper = model_helper(self.model, self.data)

    def generate_and_save_images(self, epoch):
        mean, logvar = self.model.encode(self.m_helper.test_sample)
        z = self.model.reparameterize(mean, logvar)
        predictions = self.model.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
    
    def run(self):
        self.generate_and_save_images(0)

        for epoch in range(1, self.m_helper.epochs + 1):
            start_time = time.time()
            for train_x in self.data.train_dataset:
                self.m_helper.train_step(train_x)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in self.data.test_dataset:
                loss(self.m_helper.compute_loss(test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, end_time - start_time))
            self.generate_and_save_images(epoch)

            return self